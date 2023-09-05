# -*- coding: utf-8 -*-
# @Date    : 2023/8/28 14:44
# @Author  : stellahong (stellahong@fuzhi.ai)
# @Desc    :
from functools import wraps
import asyncio


from metagpt.logs import logger
from metagpt.roles.ui_designer import Designer
from metagpt.schema import Message
from metagpt.actions.ui_design import ModelSelection
from metagpt.actions.fashion_photos import FashionPhotoPromptSelection, SDPromptImprove, FashionPhotoGeneration



def retrieve(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        content, keyword = func(*args, **kwargs)
        info = content.replace(keyword, "")
        return info
    
    return wrapper


class FashionPhotographer(Designer):
    def __init__(self, name="Richard",
                 profile="a masterful photographer with an eye for elegance",
                 goal="Generate a beautiful portrait",
                 constraints="It is important to prioritize portrait generation and maintain a professional approach."):
        super().__init__(name, profile, goal, constraints, actions=[FashionPhotoPromptSelection, FashionPhotoGeneration])
    
    async def _think(self) -> None:
        if self._rc.todo is None:
            self._set_state(0)
            return
        
        if self._rc.state + 1 < len(self._states):
            self._set_state(self._rc.state + 1)
        else:
            self._rc.todo = None
    
    async def handle_prompt_selection(self, query):
        photo_spec = FashionPhotoPromptSelection()
        self.memory_property(self.memory_user_input, query)
        template_prompt, template_id = await photo_spec.run(query)
        logger.info(template_id)
        logger.info(template_prompt)
        if int(template_id) == -1:
            ms = ModelSelection()
            model_name, domain = await ms.run(query)
            logger.info(f"{model_name}, {domain}")
    
            sd_pi = SDPromptImprove()
            resp = await sd_pi.run(query=query, domain=domain, answer_count=1)
            logger.info(resp)
            template_prompt = resp[0]

        else:
            logger.info(template_prompt)

        return template_prompt
        
 
    async def handle_sd_generation(self, *args):
        msg = self._rc.memory.get_by_action(FashionPhotoPromptSelection)[0]
        image_name = self.get_important_memory(self.memory_user_input)
        template_prompt = msg.content.replace(self.profile+":","")
        logger.info(template_prompt)
        model_name = "全网首发｜TWing shadow光影摄影V1.1_v1.2"
        kwargs = {"height": 768}
        
        generate_action = FashionPhotoGeneration()
        prompts = [generate_action._construct_prompt(template_prompt, model_name, **kwargs)]
        await generate_action.engine.run_t2i(prompts, save_name=image_name)
        return msg
    
    async def _act(self) -> Message:
        logger.info(f"{self._setting}: ready to {self._rc.todo}")
        todo = self._rc.todo
        msg = self._rc.memory.get(k=1)[0]
        query = msg.content
        logger.info(msg.cause_by)
        logger.info(query)
        logger.info(todo)
        
        handler_map = {
            
            FashionPhotoPromptSelection: self.handle_prompt_selection,
            FashionPhotoGeneration: self.handle_sd_generation,
        }
        handler = handler_map.get(type(todo))
        if handler:
            resp = await handler(query)
            
            ret = Message(f"{resp}", role=self.profile, cause_by=type(todo))
            self._rc.memory.add(ret)
            return ret
        
        raise ValueError(f"Unknown todo type: {type(todo)}")
    
    async def _react(self) -> Message:
        while True:
            await self._think()
            if self._rc.todo is None:
                break
            
            msg = await self._act()
        return msg


if __name__ == "__main__":
    for query in ["胶片风格", "气质美女", "复古写真"]:
        role = FashionPhotographer()
        asyncio.run(role.run(query))
        