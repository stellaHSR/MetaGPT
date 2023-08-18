# -*- coding: utf-8 -*-
# @Date    : 2023/8/17 13:43
# @Author  : stellahong (stellahong@fuzhi.ai)
# @Desc    :
from __future__ import annotations

import json
from typing import Callable, Any, List

from metagpt.actions import Action
from metagpt.logs import logger
from metagpt.tools.sd_engine import SDEngine
from metagpt.utils.common import OutputParser

MODEL_SELECTION_PROMPT = """Please help me find a suitable model for painting in this scene.
Model list will be given in the format like:
'''
model_name: model desc,
'''

you should select the model and tell me the model name. answer it in the form like Model: model_name || Domain:xxx

###
Model List:
{model_info}

My scene is: {query}
"""

DOMAIN_JUDGEMENT_TEMPLATE = '''
use model {model_name}, deside the domain, answer it in the form like Domain: xxx

###
Model Information:
{model_info}

'''

MODEL_SELECTION_OUTPUT_MAPPING = {
    "Model:": (str, ...), }

SD_PROMPT_KW_OPTIMIZE_TEMPLATE = '''
I want you to act as a prompt generator. Compose each answer as a visual sentence. Do not write explanations on replies. Format the answers as javascript json arrays with a single string per answer. Return exactly {answer_count} to my question. Answer the questions exactly, in the form like responses:xxx. Answer the following question:

Find 3 keywords related to the prompt "{messages}" that are not found in the prompt. The keywords should be related to each other. Each keyword is a single word.

'''

SD_PROMPT_IMPROVE_OPTIMIZE_TEMPLATE = '''
I want you to act as a prompt generator. Compose each answer as a visual sentence. Do not write explanations on replies. Format the answers as javascript json arrays with a single string per answer. Return exactly {answer_count} to my question. Answer the questions exactly, in the form like responses:xxx. Answer the following question:

domain is {domain}

if domain is anime or game like,  Take the prompt "{messages}, Cute kawaii sticker , white background,  vector, pastel colors" and improve it.

if domain is realistic like, Take the prompt "{messages}" and improve it.

'''
# Die-cut sticker, illustration minimalism,

FORMAT_INSTRUCTIONS = """The problem is to make the user input a better text2image prompt, the input is {query}"

    Let's first understand the problem and devise a plan to solve the problem.

    Based on the text2image model selected {model_name} and domain {domain}
    You have access to the following tools:

    {tool_names}
    {tool_description}

    Use a json blob to specify a tool by providing an action key (tool name) and an Observation (tool description).

    Valid "action" values: {tool_names}

    Provide only ONE action per $JSON_BLOB, as shown:

    ```
    {{{{
      "action": $TOOL_NAME,
      "Observation": $TOOL_DESCRIPTION
    }}}}
    ```

    Follow this format:

    ## Think Chain
    ```
    Question: input question to answer
    Thought: select a better method for the input by go through these two tools and its observations respectively
    Action1:
    ```
    $JSON_BLOB
    ```
    Action2:
    ```
    $JSON_BLOB
    ```

    Thought: Evaluate a better a prompt by considering the prompt richness, and you can only select one tool
    ```
    
    Finish this selection, in the form:
    ## Final Action:
    tool name
    
    """

PROMPT_OUTPUT_MAPPING = {
    "Final Action:": (str, ...),
}


class ModelSelection(Action):
    def __init__(self, name: str = "", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.desc = "Select models"
    
    def add_models(self, model_name="", model_desc=""):
        """
        fixme: 添加已有模型列表 & 模型信息
        """
        default_model_info = {"realisticVisionV30_v30VAE": "Real Effects, Real Photo/Photography, v3.0",
                              "pixelmix_v10": "an anime model merge with finetuned lineart and eyes."}
        if len(model_name) == 0:
            return default_model_info
        
        default_model_info.update({model_name: model_desc})
        return default_model_info
    
    async def run(self, query, system_text: str = "model selection"):
        prompt = MODEL_SELECTION_PROMPT.format(query=query, model_info=self.add_models())
        logger.info(prompt)
        resp = await self._aask(prompt=prompt, system_msgs=[system_text])
        logger.info(resp)
        result = resp.split("||")
        model_name = result[0].replace("Model:", "").strip()
        logger.info(model_name)
        domain = result[-1].replace("Domain:", "").strip()
        return model_name, domain


class LoraSelection(Action):
    """
    筛选SD适配当前模型和需求的可用lora，如人物可用亚洲、欧美、日韩多个lora，选择最合适的Lora
    """
    
    def __init__(self, name: str = "", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.desc = ""
    
    async def run(self, *args, **kwargs):
        # ".<lora:WZ0710_AW81e-3_30e3b128d64T32_goon0.5>"
        pass


class SDGeneration(Action):
    def __init__(self, name: str = "", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.desc = ""
        self.engine = SDEngine()
        self.negative_prompts = {"realisticVisionV30_v30VAE": "worst quality, low quality, easynegative",
                                "pixelmix_v10": ""}  #
    
    async def run(self, query, model_name, **kwargs):
        """
        generate image via sd t2i API
        """
        img_name = kwargs.get("image_name", "")
        negative_prompt = self.negative_prompts[model_name]
        if isinstance(query, str):
            prompt = self.engine.construct_payload(query, negative_prompt=negative_prompt, sd_model=model_name)
            await self.engine.run_t2i([prompt], save_name=img_name)
        elif isinstance(query, list):
            prompts_batch = []
            for _p in query:
                prompt = self.engine.construct_payload(_p, negative_prompt=negative_prompt, sd_model=model_name)
                prompts_batch.append(prompt)
            await self.engine.run_t2i(prompts_batch, save_name=img_name)


class SDPromptOptimize(Action):
    """
    扩充画图的提示词，根据keyword
    """
    
    def __init__(self, name: str = "", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.desc = ""
    
    async def run(self, query, domain: str = "realistic", answer_count=1):
        system_primer = f"Act like you are a terminal and always format your response as json. Always return exactly {answer_count} anwsers per question."
        prompt = SD_PROMPT_KW_OPTIMIZE_TEMPLATE.format(messages=query, answer_count=answer_count)
        resp: str = await self._aask(prompt=prompt, system_msgs=[system_primer])
        try:
            resp = json.loads(resp)
            responses = resp["responses"]
            logger.info(responses)
            return responses
        except:
            logger.error("Invalid LLM resp!")
            return [query]


class SDPromptImprove(Action):
    """
    提升输入的提示词（建议当输入的提示词较长时）
    fixme: 接入提示词优化的FT模型
    """
    
    def __init__(self, name: str = "", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.desc = ""
    
    async def run(self, query, domain: str = "realistic", answer_count=1):
        system_primer = f"Act like you are a terminal and always format your response as json. Always return exactly {answer_count} anwsers per question."
        prompt = SD_PROMPT_IMPROVE_OPTIMIZE_TEMPLATE.format(messages=query, domain=domain, answer_count=answer_count)
        resp: str = await self._aask(prompt=prompt, system_msgs=[system_primer])
        try:
            resp = json.loads(resp)
            responses = resp["responses"]
            logger.info(responses)
            return responses
        except:
            logger.error("Invalid LLM resp!")
            return [query]


class SDPromptExtend(Action):
    def __init__(self, name: str = "", tools: List = [], **kwargs):
        super().__init__(name, **kwargs)
        self.desc = ""
        self.tools = tools  # 基于prompt engineering 的tools
        logger.info(self.tools)
    
    def _parse_tools(self):
        tool_strings = []
        for tool in self.tools:
            tool_strings.append(f"{tool.name}: {tool.description}")
        
        formatted_tools = "\n".join(tool_strings)
        tool_names = ", ".join([tool.name for tool in self.tools])
        return tool_names, formatted_tools
    
    async def run(self, query, answer_count=1, domain: str = "realistic", model_name="realisticVisionV30_v30VAE"):
        tool_names, formatted_tools = self._parse_tools()
        msg = FORMAT_INSTRUCTIONS.format(query=query, tool_names=tool_names,
                                         tool_description=formatted_tools,
                                         model_name=model_name, domain=domain)
        
        resp = await self._aask(msg)
        
        logger.info(resp)
        output_block = OutputParser.parse_data_with_mapping(resp, PROMPT_OUTPUT_MAPPING)
        selection = output_block["Final Action"]
        
        return selection


class SDPromptRanker(Action):
    """
    根据多个Prompts进行打分和排序，选出最适合当前需求和基础模型的prompt
    """
    
    def __init__(self, name: str = "", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.desc = ""
    
    async def run(self, *args, **kwargs):
        """
        select the best prompt according to the model/requirement
        """
        
        pass


class SDImgScorer(Action):
    """
    根据多个SD的生成结果，进行美学评分，选出评分最高的图片
    """
    
    def __init__(self, name: str = "", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.desc = ""
    
    async def run(self, *args, **kwargs):
        pass


class Tool:
    def __init__(self, name: str, func: Callable, description: str, **kwargs: Any) -> None:
        """Initialize tool."""
        self.name = name
        self.func = func
        self.description = description


tools = [
    Tool(name="PromptOptimize",
         func=SDPromptOptimize().run,
         description="Find 3 keywords related to the prompt  that are not found in the prompt. The keywords should be related to each other. Each keyword is a single word. useful for when you need to add extra keywords for input prompt, specially for long enough input"),
    
    Tool(name="PromptImprove",
         func=SDPromptImprove().run,
         description="Take the prompt and improve it. useful for when you need to add improve and extend the prompt for input prompt, specially for short input"),

]

if __name__ == "__main__":
    import asyncio
    
    query = "Pokémon Go"
    
    # sd_exd = SDPromptExtend(tools=tools)
    # resp = asyncio.run(sd_exd.run(query))
    # logger.info(resp)
    
    ms = ModelSelection()
    model_name, domain = asyncio.run(ms.run(query))
    logger.info(model_name)
    sd_po = SDPromptOptimize()
    sd_pi = SDPromptImprove()
    resp = asyncio.run(sd_pi.run(query=query, domain=domain, answer_count=1))
    sd_gen = SDGeneration()
    resp = asyncio.run(sd_gen.run(query=resp, model_name=model_name))
    logger.info(resp)
