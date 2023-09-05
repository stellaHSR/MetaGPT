# -*- coding: utf-8 -*-
# @Date    : 2023/8/28 17:05
# @Author  : stellahong (stellahong@fuzhi.ai)
# @Desc    :
import json
from typing import List, Optional, Tuple, Union

import requests

from metagpt.logs import logger
from metagpt.actions.ui_design import SDGeneration
from metagpt.actions.design import BaseModelAction
from metagpt.actions.ui_design import ModelSelection
from metagpt.actions.design import SDPromptImprove

DEFAULT_NEGATIVE_PROMPTS = "(ng_deepnegative_v1_75t:1.1),negative_hand-neg," \
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

DEFAULT_TEMPLATES = {
    "dreamy portrait, natural 35mm film, sunlit whimsy": "<lora:FilmVelvia3:0.6>, Best portrait photography, 35mm film, natural blurry, 1girl, sun dress, wide brimmed hat, radiant complexion, whimsical pose, fluttering hair, golden sunlight, macro shot, shallow depth of field, bokeh, dreamy",
    "Classic Hollywood Glamour, Girl in a stunning red evening gown, Portrait photography, natural blurriness": "<lora:FilmVelvia3:0.8>, Classic Hollywood glamour, girl in a stunning pink evening gown, dazzling accessories, (captivating eyes, red lips, flawless skin), luxurious backdrop, velvet curtains, dramatic pose, interesting composition, spotlight illumination, rich shadows, shallow depth of field, sparkling bokeh, exquisite details, high-resolution, 35mm film, timeless elegance. portrait photography, 35mm film, natural blurry",
    "Hyperrealistic Fashion Portrait, Portrait of a girl, Off-shoulder outfit, necklace, Fluffy short hair": "<lora:FilmVelvia3:0.6>, fashion photography portrait of 1girl, offshoulder, fluffy short hair, soft light, rim light, beautiful shadow, low key, (photorealistic, raw photo:1.2), (natural skin texture, realistic eye and face details), hyperrealism, ultra high res, 4K, Best quality, masterpiece, necklace, (cleavage:0.8), in the dark",
    "Cinematic Elegance Portrait, Elegant portrayal of a girl, Forest setting, ": "<lora:FilmVelvia3:0.6>, Best portrait photography, RAW photo, 8K UHD, film grain, cinematic lighting, elegant 1girl, offshoulder dress, (natural skin texture), intricate (fishnet stockings), forest, natural pose, embellishments, cinematic texture, 35mm lens, shallow depth of field, silky long hair",
    "Candy Fantasy Portrait, Sweet, playful, and whimsical interpretation, ": "<lora:FilmVelvia3:0.6>, The model's bust is composed entirely of pieces of multi-colored hard candy stacked together. Her facial features are stylized with different flavors and colors of candy - red lips, blue eyes, yellow hair. A few pieces come loose and dangle down, about to drop off. It's a sweet, playful and whimsical interpretation of her portrait. (RAW photo, best quality),(realistic, photo-realistic:1.3), clean, masterpiece,finely detail,masterpiece,ultra-detailed,highres,(best illustration),(best shadow),intricate, bright light",
    "Japanese model, floral kimono, captivating gaze, intricate hair ornament, outdoor elegance": "<lora:FilmVelvia3:0.6>, 1girl, solo, outdoor, cute japanese model girl, kimono, floral print, hair ornament, looking at viewer, hair flower, brown eyes, bangs, masterpiece, best quality, realistic",
    "Winter charm, school uniform, snowy backdrop, enchanting smile, outdoor elegance": "<lora:FilmVelvia3:0.6>, 1girl, black hair, bag, solo, long hair, snow, looking at viewer, tree, outdoors, winter, necktie, smile, coat, shoulder bag, school bag, black eyes, bare tree, school uniform, lips, blue necktie, masterpiece, best quality, realistic",
    "Ancient Chinese clothing, winter,Chinese style, Dream of the Red Chamber":"1 girl, cowboy shot, <lora:Winter_Hanfu:0.95>, (Winter hanfu), (cloak, headwear), (snow, white flowers garden:1.2), ",
    "Classical Hanfu, Ming Dynasty style, Hanfu fashion":"(8k, best quality, masterpiece:1.2), (realistic, photo-realistic:1.2), Best portrait photography, 1girl,perfect face, perfect eyes,pureerosface_v1, blue hanfu, ming style,(full body:1.2),<lora:hanfu_v30:0.75>,",
    "wedding dress, wedding photograph, veil, white dress":"Best Quality, Masterpiece:1.4),(Realism:1.2),(Realisitc:1.2),(Absurdres:1.2),4K,(photorealistic:1.3), Detailed, Ultra-Detailed, Digital Art, beautiful face, beautiful detailed eyes, closed mouth, symmetric eyes,1girl, Solo,(Long Wavy Hair),<lora:2023-05-14_120014:0.8>, hunsha, (veil:1.2), Earrings, Jewelry, SunLight, Flowers, Outdoors, Hair Ornaments, White Flowers, Vast Open Field, (bright light),",
    "Chinese wedding dress, Chinese clothes, red dress, ancient cloth, ":"Best Quality, Masterpiece:1.4),(Realism:1.2),(Realisitc:1.2),(Absurdres:1.2),4K,(photorealistic:1.3), Detailed, Ultra-Detailed, a woman in a red and gold dress, Phoenix crown, hair stick, upper body, portrait, sit on a red bed, solo, looking at viewer,  long hair, Chinese clothes, hair ornament, dress, smile, realistic,  red lips, blurry, indoors, red dress, closed mouth, blurry background, jewelry, lips, black eyes, black hair, sunlight, (Red quilt),(red palace:1.2),(ancient Chinese architecture),(red:1.8),<lora:fgxp:0.8>",
}

# '''
# scene_tag: scene desc,
# '''

TEMPLATES_SELECTION_PROMPT = """
Role: You are a professional photographer; the goal is to select the suitable scene and style according to the user requirement
Please help me find a suitable scene and style for painting in this scene.
Scenes and styles list will be given in the format like:
'''
scene_tag_id: scene_tag,
'''

you should select the scene and tell me the scene_tag. answer it in the form like Answer: scene_tag_id || scene_tag

###
Scenes List:
{scene_info}

My requirement is: {query}

"""


class FashionPhotoPromptSelection(BaseModelAction):
    def __init__(self, name: str = "", *args, **kwargs):
        super().__init__(name, description="FilmStylePhotoPromptSelection", **kwargs)
        self.DEFAULT_TEMPLATES = DEFAULT_TEMPLATES
    
    def add_prompt_templates(self, tag="", template=""):
        updated_info = {tag: template} if tag else {}
        templates = {**self.DEFAULT_TEMPLATES, **updated_info}
        scene_tags = {id: key for id, (key, val) in enumerate(templates.items())}
        return scene_tags
        
    
    async def run(self, query: str, system_text: str = "prompts selection"):
        prompt = TEMPLATES_SELECTION_PROMPT.format(query=query, scene_info=self.add_prompt_templates())
        resp = await self._aask(prompt=prompt, system_msgs=[])
        
        logger.info(resp)
        result = resp.split("||")
        try:
            template_id = int(result[0].replace("Answer:", "").strip())
            template_tag = result[1].strip()
        except:
            logger.error("invalid scene prompts!")
            template_id = -1
            return  query,  template_id
        
        try:
            template_prompt = self.DEFAULT_TEMPLATES[template_tag]
        except:
            template_prompt = list(self.DEFAULT_TEMPLATES.values())[template_id]
        
        return template_prompt, template_id


class FashionPhotoGeneration(SDGeneration):
    """ 和SD默认调用不同，这里配置写真照使用的不同底模+负提示词配置，以及需要初始化SD入参配置模板"""
    
    def __init__(self, name: str = "", description="FilmStylePhotoGeneration", *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.negative_prompts = {
            "realisticVisionV30_v30VAE": DEFAULT_NEGATIVE_PROMPTS,
            "galaxytimemachinesGTM_photoV20": DEFAULT_NEGATIVE_PROMPTS,
            "majicMIX realistic 麦橘写实_v2威力加强典藏版": DEFAULT_NEGATIVE_PROMPTS,
            "全网首发｜TWing shadow光影摄影V1.1_v1.2": DEFAULT_NEGATIVE_PROMPTS,
            "majicMIX realistic 麦橘写实_v6": DEFAULT_NEGATIVE_PROMPTS,
            
        }
        
        self.engine.payload["cfg_scale"] = 7
        self.engine.payload["sampler_index"] = "DPM++ 2M SDE Karras"
        
    
    def _construct_prompt(self, query: str, model_name: str = "全网首发｜TWing shadow光影摄影V1.1_v1.2", **kwargs) -> str:
        """Constructs a prompt for the provided query and model."""
        negative_prompt = self.negative_prompts.get(model_name, DEFAULT_NEGATIVE_PROMPTS)
        height = kwargs.get("height", 768)
        return self.engine.construct_payload(query, negative_prompt=negative_prompt, height=height, sd_model=model_name,
                                             restore_faces=True)


