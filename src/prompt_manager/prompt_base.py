import logging
from abc import ABC, abstractmethod
import re

class PromptBaseMethod(ABC):
    def __init__(self, args):
        self.args = args
        logging.info(f"model_name: {args.model}, model_id: {args.model_id}")
        self.model = self.args.model_class(self.args)

    def parse_output_for_prompt_method(self, output):
        pattern = r'\[(.*?)\]'
        match = re.search(pattern, output)
        if match:
            return match.group(1)
        return output
    
    def run(self, prompt, input):

        cur_prompt = prompt.format(input=input)
        # return cur_prompt
        
        resp, total_tokens, duration = self.model.generate(cur_prompt)

        return resp, total_tokens, duration

        if not self.args.prompt_method:
            resp = resp.split("\n")[0].split("#")[0].strip()
            return resp, total_tokens, duration
        else:
            resp = self.parse_output_for_prompt_method(resp)
            return resp, total_tokens, duration

