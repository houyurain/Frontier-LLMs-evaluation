import os
import logging
import time
import torch
# from time import sleep
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import GPTQConfig, AwqConfig
from .quantization import get_model_kwargs

class ModelBase:
    def __init__(self, args):
        self.args = args
        self.tokenizer = None
        self.model_name = args.model
        self.model_id = args.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            use_fast=True,
            # padding_side="left",
        )
        self.model = self._init_model()
        self.device = next(self.model.parameters()).device
        
        
    def _init_model(self):
        """Initialize the model with optional quantization"""
        model_kwargs = get_model_kwargs(self.args)
        logging.info(f"Model kwargs: {model_kwargs}")
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        logging.info(f"Model loaded successfully with quantization: {getattr(self.args, 'quantization', 'none')}")
        return model

    def inference(self, prompt):
        
        input_ids = self.tokenizer(
            prompt, 
            return_tensors="pt",
        ).to(self.device)

        start_time = time.time()       

        outputs  =  self.model.generate(
            **input_ids,
            max_new_tokens=self.args.max_new_tokens,
            temperature=self.args.temperature
        )

        end_time = time.time()

        response = self.tokenizer.decode(outputs[0][input_ids["input_ids"].shape[-1]:], skip_special_tokens=True) 

        return response, outputs.shape[-1], end_time - start_time

    def generate(self, prompt) -> str:
        """unified generation interface"""
        
        resp, total_tokens, duration = self.inference(prompt)
        resp = resp.lstrip('\n')
        logging.info(f"Response: \n{resp}")
        # resp = resp.split("\n")[0].split("#")[0].strip()

        return resp, total_tokens, duration



if __name__ == "__main__":
    # Example usage
    class Args:
        model = "gpt2"
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        quantization = "8bit"  # Example quantization setting
        max_new_tokens = 1024
        temperature = 0.7

    args = Args()
    model_base = ModelBase(args)
    print(model_base.model)  # Print the model to verify initialization
    print(model_base.generate("Hello, how are you?"))  # Example generation call)