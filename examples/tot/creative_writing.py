# -*- coding: utf-8 -*-
# @Date    : 12/25/2023 1:06 PM
# @Author  : stellahong (stellahong@fuzhi.ai)
# @Desc    :
import re

from examples.tot.prompt_templates.creative_writing import cot_prompt, vote_prompt
from metagpt.strategy.tot import TreeofThought
from metagpt.strategy.tot_schema import (
    BaseEvaluator,
    BaseParser,
    Strategy,
    ThoughtSolverConfig,
)


class TextGenParser(BaseParser):
    propose_prompt: str = cot_prompt
    value_prompt: str = vote_prompt

    def __call__(self, input_text: str) -> str:
        return input_text

    def propose(self, current_state: str, **kwargs) -> str:
        return self.propose_prompt.format(input=current_state, **kwargs)

    def value(self, input: str = "", **kwargs) -> str:
        # node_result = self(input)
        id = kwargs.get("node_id", "0")
        return self.value_prompt + f"Choice {id}:\n{input}\n"


class TextGenEvaluator(BaseEvaluator):
    value_map: dict = {"impossible": 0.001, "likely": 1, "sure": 20}  # TODO: ad hoc
    status_map: dict = {val: key for key, val in value_map.items()}

    def __call__(self, evaluation: str, **kwargs) -> float:
        try:
            value = 0
            node_id = kwargs.get("node_id", "0")
            pattern = r".*best choice is .*(\d+).*"
            match = re.match(pattern, evaluation, re.DOTALL)

            if match:
                vote = int(match.groups()[0])
                print(vote)
                if vote == int(node_id):
                    value = 1
        except:
            value = 0
        return value

    def status_verify(self, value):
        status = False
        if value in self.status_map:
            status_value = self.status_map[value]
            if status_value != "impossible":
                status = True
        return status


if __name__ == "__main__":
    import asyncio

    initial_prompt = """It isn't difficult to do a handstand if you just stand on your hands. It caught him off guard that space smelled of seared steak. When she didn’t like a guy who was trying to pick her up, she started using sign language. Each person who knows you has a different perception of who you are."""

    parser = TextGenParser()
    evaluator = TextGenEvaluator()

    config = ThoughtSolverConfig(n_generate_sample=3, parser=parser, evaluator=evaluator)

    tot_base = TreeofThought(strategy=Strategy.BFS, config=config)
    asyncio.run(tot_base.solve(init_prompt=initial_prompt))
