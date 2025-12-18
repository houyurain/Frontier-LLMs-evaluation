from .model_base import ModelBase
import time
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from .quantization import get_model_kwargs


class GemmaModel(ModelBase):

    def _init_model(self):
        """Initialize the model with optional quantization"""
        model_kwargs = get_model_kwargs(self.args)
        logging.info(f"Model kwargs: {model_kwargs}")
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        # model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        logging.info(f"Model loaded successfully with quantization: {getattr(self.args, 'quantization', 'none')}")
        return model

    def inference(self, prompt):
        """Override inference for LlamaModel if needed"""
        # prompt = "Write me a poem about Machine Learning."
        # messages = [
        #     {"role": "user", "content": prompt},
        # ]

        # input_ids = self.tokenizer.apply_chat_template(
        #     messages,
        #     return_tensors="pt",
        #     return_dict=True,
        # ).to(self.device)

        input_ids = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        start = time.time()
        outputs  =  self.model.generate(
            **input_ids,
            max_new_tokens=self.args.max_new_tokens,
            temperature=self.args.temperature
        )
        end = time.time()
        duration = end - start
        logging.info(outputs)

        total_tokens = outputs.shape[-1]  # Total tokens = output tokens + input tokens

        response = self.tokenizer.decode(outputs[0][input_ids["input_ids"].shape[-1]:]) #

        return response, total_tokens, duration








