
from transformers import BitsAndBytesConfig, GPTQConfig, AwqConfig
import torch
import logging

def get_model_kwargs(args, tokenizer=None):
    """
    Prepare model kwargs based on quantization settings.
    This function sets up the model configuration for loading with or without quantization.
    """

    quantization_config = None
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
        
    # Check if quantization is specified
    if hasattr(args, 'quantization') and args.quantization:
        quantization_method = args.quantization.lower()
            
        if quantization_method == "4bit":
            # 4-bit quantization using BitsAndBytes
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            logging.info("===Using 4-bit quantization")
            
        elif quantization_method == "8bit":
            # 8-bit quantization using BitsAndBytes
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            logging.info("===Using 8-bit quantization")
        else:
            # Default: no quantization, use float16
            model_kwargs["torch_dtype"] = torch.float16
            logging.info("===No quantization specified, using float16")
            
    else:
        raise ValueError(f"Unsupported quantization method: {quantization_method}")
    
    
    # Add quantization config if specified
    if quantization_config:
        model_kwargs["quantization_config"] = quantization_config
    
    return model_kwargs