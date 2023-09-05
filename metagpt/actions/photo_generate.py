# -*- coding: utf-8 -*-
# @Date    : 2023/8/19 17:05
# @Author  : stellahong (stellahong@fuzhi.ai)
# @Desc    :
from typing import List

from metagpt.logs import logger
from metagpt.actions.design import SDPromptImprove
from metagpt.actions.ui_design import SDGeneration
from metagpt.actions.design import BaseModelAction

ID_PHOTO_PROMPT = """
I want you to act as a prompt generator. Compose each answer as a visual sentence. Do not write explanations on replies. Format the answers as javascript json arrays with a single string per answer. Return exactly {answer_count} to my question. Answer the questions exactly, in the form like responses:xxx. Answer the following question:

Take the prompt "{messages}" and improve it.

Here are good examples:
1. A man posing for a portrait photograph, donned in a vibrant red suit against a pristine (white background)
2. A female model posing for a portrait photography, elegantly dressed in a vibrant red suit against a pristine (white background)
"""

#
# Here is an example you can refer to: , and give a general description.

ID_PHOTO_VERIFY_PROMPT = """
As a professional ID photo photographer, you need to judge the type of photo I want to take based on my input: {messages}.
Please provide the photo with the following specifications background color, cloth requirements

Provide only ONE action per $JSON, as shown:

    
{{{{
      "user input": $message,
      "specification": portrait photography, $(background-color) wearing $(color) cloth, $(gender),$(nationality),$(hair)
}}}}

Format the answers as javascript json arrays with a single string per answer. Return exactly one response to my question.
"""


class IDPhotoPromptOptimize(SDPromptImprove):
    def __init__(self, name: str = "", description="IDPhotoPromptOptimize", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
    
    async def run(self, query: str, domain: str = "ID Photo", answer_count: int = 1) -> List[str]:
        """Run the improvement for the given query."""
        prompt_prefix = "(((best quality))),(((ultra detailed))),(((masterpiece))),"
        prompt_postfix = ",identification photo,<lora:identification_photo:1>"
        resp = await self.run_optimize_or_improve(query, domain, ID_PHOTO_PROMPT,
                                                  answer_count=answer_count)
        improved_prompts = [prompt_prefix + prompt + prompt_postfix for prompt in resp]
        return improved_prompts


class IDPhotoSDGeneration(SDGeneration):
    """ 和SD默认调用不同，这里配置写真照使用的不同底模+负提示词配置，以及需要初始化SD入参配置模板"""
    
    def __init__(self, name: str = "", description="IDPhotoSDGeneration", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.negative_prompts = {
            "realisticVisionV30_v30VAE": "(ng_deepnegative_v1_75t:1.1),negative_hand-neg," \
                                         "verybadimagenegative_v1.3,kkw-ph1-neg,Asian-Less-Neg,canvas frame,cartoon,3d,((disfigured)),((bad art)),((deformed))," \
                                         "((extra limbs)),((close up)),((b&w)),blurry,(((duplicate))),((morbid)),((mutilated))," \
                                         "[out of frame],extra fingers,mutated hands,((poorly drawn hands)),((poorly drawn face))," \
                                         "(((mutation))),(((deformed))),((ugly)),blurry,((bad anatomy)),(((bad proportions)))," \
                                         "((extra limbs)),cloned face,(((disfigured))),out of frame,ugly,extra limbs,(bad anatomy)," \
                                         "gross proportions,(malformed limbs),((missing arms)),((missing legs)),(((extra arms)))," \
                                         "(((extra legs))),mutated hands,(fused fingers),(too many fingers),(((long neck)))," \
                                         "Photoshop,video game,ugly,tiling,poorly drawn hands,poorly drawn feet," \
                                         "poorly drawn face,out of frame,mutation,mutated,extra limbs,extra legs,extra arms," \
                                         "disfigured,deformed,cross-eye,body out of frame,blurry,bad art,bad anatomy,3d render,(tiara),(crown),(badhands),",
           
            "galaxytimemachinesGTM_photoV20": "(ng_deepnegative_v1_75t:1.1),negative_hand-neg," \
                                              "verybadimagenegative_v1.3,kkw-ph1-neg,Asian-Less-Neg,canvas frame,cartoon,3d,((disfigured)),((bad art)),((deformed))," \
                                              "((extra limbs)),((close up)),((b&w)),blurry,(((duplicate))),((morbid)),((mutilated))," \
                                              "[out of frame],extra fingers,mutated hands,((poorly drawn hands)),((poorly drawn face))," \
                                              "(((mutation))),(((deformed))),((ugly)),blurry,((bad anatomy)),(((bad proportions)))," \
                                              "((extra limbs)),cloned face,(((disfigured))),out of frame,ugly,extra limbs,(bad anatomy)," \
                                              "gross proportions,(malformed limbs),((missing arms)),((missing legs)),(((extra arms)))," \
                                              "(((extra legs))),mutated hands,(fused fingers),(too many fingers),(((long neck)))," \
                                              "Photoshop,video game,ugly,tiling,poorly drawn hands,poorly drawn feet," \
                                              "poorly drawn face,out of frame,mutation,mutated,extra limbs,extra legs,extra arms," \
                                              "disfigured,deformed,cross-eye,body out of frame,blurry,bad art,bad anatomy,3d render,(tiara),(crown),(badhands),"
            
            # "worst quality, low quality, easynegative",
            # "realisticVisionV30_v30VAE": "Blurry female ID photo in a white shirt and a black suit, disheveled appearance, low resolution, lack of realism, chaotic background, watermark obstructing, wrinkled clothing, expressionless, distorted colors, poor-quality shot, unfocused, distorted facial features, amateurish photography, heavy shadows, unnatural colors, subpar image quality, unclear and blurred."
        }
        
        self.engine.payload["cfg_scale"] = 7
    
    def _construct_prompt(self, query: str, model_name: str) -> str:
        """Constructs a prompt for the provided query and model."""
        negative_prompt = self.negative_prompts.get(model_name, "")
        return self.engine.construct_payload(query, negative_prompt=negative_prompt, height=768, sd_model=model_name,
                                             restore_faces=True)


class IDPhotoCheck(BaseModelAction):
    def __init__(self, name: str = "", *args, **kwargs):
        super().__init__(name, description="define and clarify the ID photo type and specification", *args, **kwargs)
    
    async def run(self, query, *args, **kwargs):
        prompt = ID_PHOTO_VERIFY_PROMPT.format(messages=query)
        logger.info(prompt)
        resp: str = await self._aask(prompt=prompt, system_msgs=["in English"])
        result = await self.handle_response(resp)
        return result or query


if __name__ == "__main__":
    import asyncio
    
    requirements = "身份证照"
    
    #
    action = IDPhotoCheck()
    print(type(action))
    resp = asyncio.run(action.run(requirements))
    print(resp[0])
    action = IDPhotoPromptOptimize()
    resp = asyncio.run(action.run(resp[0]))
