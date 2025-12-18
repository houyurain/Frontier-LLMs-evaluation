from .model_base import ModelBase
import time
import logging

class QwenModel(ModelBase):

    def inference(self, prompt):
        """Override inference for LlamaModel if needed"""
        if 'qwen3' in self.model_name.lower():
            return self.inference_qwen3(prompt)

        messages = [
            # {"role": "system", "content": "You are a chatbot who always responds user queries in a concise manner."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        start = time.time()
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.args.max_new_tokens,
            temperature=self.args.temperature,
        )
        end = time.time()

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        total_tokens = generated_ids[0].shape[-1]  # Total tokens = output tokens + input tokens

        return response, total_tokens, end-start
    


    def inference_qwen3(self, prompt):

        messages = [
            {"role": "system", "content": "You are a chatbot who always responds user queries in a concise manner."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False, # Switches between thinking and non-thinking modes. Default is True.
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

        # conduct text completion
        start = time.time()
        generated_ids = self.model.generate(
            **model_inputs,
            temperature=self.args.temperature,
            max_new_tokens=self.args.max_new_tokens, #max_new_tokens=4096, #32768,
        )
        end = time.time()
        total_tokens = generated_ids[0].shape[-1]  # Total tokens = output tokens + input tokens
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        # print("thinking content:", thinking_content)
        # print("content:", content)

        total_tokens = generated_ids[0].shape[-1]  # Total tokens = output tokens + input tokens

        return content, total_tokens, end-start











