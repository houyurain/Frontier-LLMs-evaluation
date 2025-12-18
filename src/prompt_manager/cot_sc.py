from collections import Counter
from .prompt_base import PromptBaseMethod
import logging


class CoT_SC(PromptBaseMethod):
    def __init__(self, args):
        super().__init__(args)
        args.cnt = 5

    def run(self, prompt, input):

        preds = []
        cot_prompt = prompt.format(input=input)
        for _ in range(self.args.cnt):
            resp, total_tokens, duration = self.model.generate(cot_prompt)
            resp = self.parse_output_for_prompt_method(resp)
            if ";" in resp:
                resp = ";".join(sorted([item.strip() for item in resp.split(";")]))
            preds.append(resp)
        
        counter = Counter(preds)
        final_pred = counter.most_common(1)[0][0]
        logging.info(f"Parsed response: \n{final_pred}")

        return final_pred, total_tokens, duration
