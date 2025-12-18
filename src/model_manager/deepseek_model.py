from .model_base import ModelBase
import time
import logging
from transformers import AutoModelForCausalLM, GenerationConfig
from .quantization import get_model_kwargs

class DeepseekModel(ModelBase):

    def _init_model(self):
        """Initialize the model with optional quantization"""
        model_kwargs = get_model_kwargs(self.args)
        logging.info(f"Model kwargs: {model_kwargs}")
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs
        )
        model.generation_config = GenerationConfig.from_pretrained(self.model_id)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id

        logging.info(f"Model loaded successfully with quantization: {getattr(self.args, 'quantization', 'none')}")
        return model


    def inference(self, prompt):
        """Override inference for LlamaModel if needed"""
        messages = [
            {"role": "user", "content": prompt}
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
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










