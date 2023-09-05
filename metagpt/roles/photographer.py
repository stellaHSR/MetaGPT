# -*- coding: utf-8 -*-
# @Date    : 2023/8/19 14:44
# @Author  : stellahong (stellahong@fuzhi.ai)
# @Desc    :
from functools import wraps
import asyncio

import json5

from metagpt.logs import logger
from metagpt.roles.ui_designer import Designer
from metagpt.schema import Message
from metagpt.actions.photo_generate import IDPhotoCheck, IDPhotoPromptOptimize, IDPhotoSDGeneration
from metagpt.actions.design import SDPromptExtend



def retrieve(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        content, keyword = func(*args, **kwargs)
        info = content.replace(keyword, "")
        return info
    
    return wrapper


class Photographer(Designer):
    def __init__(self, name="Ethan",
                 profile="a masterful photographer with an eye for elegance",
                 goal="Generate a beautiful portrait",
                 constraints="It is important to prioritize portrait generation and maintain a professional approach."):
        super().__init__(name, profile, goal, constraints, actions=[IDPhotoCheck, IDPhotoPromptOptimize, IDPhotoSDGeneration])
    
    async def _think(self) -> None:
        if self._rc.todo is None:
            self._set_state(0)
            return
        
        if self._rc.state + 1 < len(self._states):
            self._set_state(self._rc.state + 1)
        else:
            self._rc.todo = None
    
    async def handle_id_photo_check(self, query):
        photo_spec = IDPhotoCheck()
        self.memory_property(self.memory_user_input, query)
        resp = await photo_spec.run(query)
        return resp
        
    async def handle_sd_prompt_improve(self, query):
        sd_imprv = IDPhotoPromptOptimize()
        
        resp = await sd_imprv.run(query=query, answer_count=1)
        return resp
    
    async def handle_sd_generation(self, *args):
        msg = self._rc.memory.get_by_action(SDPromptExtend)[0]
        image_name = self.get_important_memory(self.memory_user_input)
        resp = json5.loads(msg.content)
        logger.info(resp)
        await IDPhotoSDGeneration().run(query=resp, model_name="realisticVisionV30_v30VAE", **{"image_name": image_name})
        return resp
    
    async def _act(self) -> Message:
        logger.info(f"{self._setting}: ready to {self._rc.todo}")
        todo = self._rc.todo
        msg = self._rc.memory.get(k=1)[0]
        query = msg.content
        logger.info(msg.cause_by)
        logger.info(query)
        logger.info(todo)
        
        handler_map = {
            
            IDPhotoCheck: self.handle_id_photo_check,
            IDPhotoPromptOptimize: self.handle_sd_prompt_improve,
            IDPhotoSDGeneration: self.handle_sd_generation,
        }
        handler = handler_map.get(type(todo))
        if handler:
            resp = await handler(query)
            
            if type(todo) in [IDPhotoPromptOptimize]:
                ret = Message(f"{resp}", role=self.profile, cause_by=SDPromptExtend)
            else:
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
    
    for query in ["身份证照", "港澳通行证", "驾驶证申请照片", "红底白衬衣", "蓝底黑色西装"]:
        role = Photographer()
        asyncio.run(role.run(query))
        