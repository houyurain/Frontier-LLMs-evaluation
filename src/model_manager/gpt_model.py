from .model_base import ModelBase
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
import time
import logging

class GPT(ModelBase):

    def __init__(self, args):
        self.args = args
        self.tokenizer = None
        self.model_name = args.model
        self.model_id = args.model_id
        self.model = self._init_model()

    def _init_model(self):
        with open('/users/8/zhan8023/gpt4api.txt', 'r') as f:
            api_key = f.read().strip()

        # 初始化客户端
        client = OpenAI(api_key=api_key)
        return client

    def inference(self, prompt):
        """Override inference for LlamaModel if needed"""

        # messages = [
        #     {"role": "user", "content": prompt},
        # ]

        start = time.time()

        response = self.model.chat.completions.create(
            model="gpt-5",  # 如果是 GPT-4.1 就写 gpt-4.1 或 gpt-4o
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        end = time.time()
        duration = end - start

        total_tokens = 1

        response = response.choices[0].message.content

        return response, total_tokens, duration
    
    def generate(self, prompt) -> str:
        """unified generation interface"""
        
        resp, total_tokens, duration = self.inference(prompt)
        resp = resp.lstrip('\n')
        # try:
        #     resp = resp.split("assistantfinal")[1].strip()  # Clean up the response
        # except:
        #     pass
        # logging.info(f"Response: \n{resp}")
        # resp = resp.split("\n")[0].split("#")[0].strip()
        logging.info(f"Response: \n{resp}")
        return resp, total_tokens, duration










