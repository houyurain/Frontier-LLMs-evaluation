from .model_base import ModelBase
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import logging

class GptossModel(ModelBase):

    def _init_model(self):
        """Initialize the model with optional quantization"""
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="auto",
        )
        return model

    def inference(self, prompt):
        """Override inference for LlamaModel if needed"""

        messages = [
            {"role": "user", "content": prompt},
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)

        start = time.time()
        outputs  =  self.model.generate(
            **inputs,
            max_new_tokens=self.args.max_new_tokens,
            temperature=self.args.temperature
        )

        end = time.time()
        duration = end - start

        total_tokens = outputs.shape[-1]  # Total tokens = output tokens + input tokens

        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        return response, total_tokens, duration
    
    def generate(self, prompt) -> str:
        """unified generation interface"""
        
        resp, total_tokens, duration = self.inference(prompt)
        resp = resp.lstrip('\n')
        try:
            resp = resp.split("assistantfinal")[1].strip()  # Clean up the response
        except:
            pass
        logging.info(f"Response: \n{resp}")
        # resp = resp.split("\n")[0].split("#")[0].strip()

        return resp, total_tokens, duration










