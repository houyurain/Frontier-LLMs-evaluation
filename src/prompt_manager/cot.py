from .prompt_base import PromptBaseMethod
import logging


class CoT(PromptBaseMethod):
    def __init__(self, args):
        super().__init__(args)
    

    def run(self, prompt, input):

        cot_prompt = prompt.format(input=input)

        resp, total_tokens, duration = self.model.generate(cot_prompt)

        resp = self.parse_output_for_prompt_method(resp)
        logging.info(f"Parsed response: \n{resp}")


        return resp, total_tokens, duration
