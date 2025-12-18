from .model_base import ModelBase
import time
import logging

class LlamaModel(ModelBase):

    def inference(self, prompt):
        """Override inference for LlamaModel if needed"""
        messages = [
            {"role": "system", "content": "You are a chatbot who always responds user queries in a concise manner."},
            {"role": "user", "content": prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.device)

        start = time.time()
        outputs  =  self.model.generate(
            input_ids,
            max_new_tokens=self.args.max_new_tokens,
            temperature=self.args.temperature
        )
        end = time.time()
        duration = end - start

        total_tokens = outputs.shape[-1]  # Total tokens = output tokens + input tokens

        response = self.tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)

        return response, total_tokens, duration










