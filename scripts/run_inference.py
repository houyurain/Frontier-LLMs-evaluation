import os
import logging
import argparse
# import dotenv
# dotenv.load_dotenv()

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import json
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import dataset_manager
from model_manager import ModelBase, LlamaModel, GemmaModel, QwenModel, DeepseekModel, GptossModel, GPT
from prompt_manager import PromptBaseMethod, CoT, CoT_SC  #, ToT, Self_Refine


def generate(args, dataset):
    # model = args.model_class(args)
    prompt_method = args.prompt_class(args)

    logging.info("=====Start generating=====")
    logging.info(f"=====Will evaluate {args.end_idx - args.start_idx} items=====")
    results = []

    fw = open(args.log_path_json, "w", encoding="utf-8")

    prompt = dataset.get_prompt()
    logging.info(f"=="*30)
    logging.info(f"Prompt: {prompt}")
    logging.info(f"=="*30)

    try:
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)  # Reset peak memory stats before generation
    except:
        pass

    for i in tqdm(range(args.start_idx, args.end_idx)):
        input, label = dataset.data['test'].iloc[i]['text'], dataset.data['test'].iloc[i]['label']
        cur_prompt = prompt.format(input=input)
        
        result_dict = {
            "custom_id": f"request-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-5", 
                "messages": [
                        {"role": "user", "content": cur_prompt}
                    ],
            },
        }
        results.append(result_dict)

        json.dump(result_dict, fw, ensure_ascii=False)
        fw.write("\n")
    fw.close()

    # with open(args.log_path_json, "w", encoding="utf-8") as fw:
    #     json.dump(results, fw, indent=4, ensure_ascii=False)
    # logging.info(f"Results saved to {args.log_path_json}")

    logging.info("=====Generation finished=====")

    return

def evaluate(args, dataset):
    logging.info("=====Start evaluating=====")
    logging.info(f"=====Will evaluate {args.end_idx - args.start_idx} items=====")
    args.log_path_json = args.log_path_json.replace('.jsonl', 'result.json')
    with open(args.log_path_json, "r", encoding="utf-8") as fr:
        data = fr.readlines()

    res_dict = {}

    res_list = []
    # res_dict = {"peak_memory":data[0]['peak_memory'], "total_tokens":0, "duration":0}
    for d in data:

        d = json.loads(d)
        index = int(d["custom_id"].split("-")[1])
        dict_ = {
            "response": d["response"]['body']['choices'][0]['message']['content'],
            "label": dataset.data['test'].iloc[index]['label'],
        }
        res_list.append(dict_)
        # print(json.dumps(d, indent=4, ensure_ascii=False))
        # exit()
        # res_dict["total_tokens"] += d["total_tokens"]
        # res_dict["duration"] += d["duration"]
    
    # res_dict["throughput"] = res_dict["total_tokens"] / res_dict["duration"]
    # res_dict["latency"] = res_dict["duration"] / (args.end_idx - args.start_idx)

    res_dict.update(dataset.evaluate(res_list))

    logging.info("=====Evaluation finished=====")
    for key, value in res_dict.items():
        logging.info(f"{key}: {value}")
    
    with open(args.result_path_json, "w", encoding="utf-8") as fw:
        json.dump(res_dict, fw, indent=4, ensure_ascii=False)
    logging.info(f"Results saved to {args.result_path_json}")

    print(res_dict)


def get_args():
    parser = argparse.ArgumentParser()
    # ===Arguments for dataset===
    parser.add_argument("--dataset", type=str, required=True)

    # ===Arguments for model and method===
    parser.add_argument("--model", type=str, required=True)
    # parser.add_argument("--method", type=str, required=True)    
    
    # ===Arguments for OPEN-SOURCE models===
    parser.add_argument("--temperature", type=float, default=0.7, help="temperature")
    parser.add_argument("--seed", type=int, default=42, help="seed for vllm inference")

    # ===Arguments for prompt===
    parser.add_argument("--few_shot", action="store_true", help="Few shot learning")
    parser.add_argument("--num_examples", type=int, default=1, help="Number of examples for few-shot setting")

    # ===Arguments for prompting method===
    parser.add_argument("--prompt_method", type=str, default=None, help="Prompting method", choices=["cot", "sc_cot"])

    parser.add_argument("--max_new_tokens", type=int, default=256, help="maximum number of newly generated tokens")
    # parser.add_argument("--tensor_parallel", type=int, default=2, help="number of GPUs to use for running open-source models only.")
    parser.add_argument("--quantization", type=str, default="none", help="quantization", choices=["4bit", "8bit", "none"])
    # parser.add_argument("--gpu_utilization", type=float, default=0.99, help="GPU utilization")

    # ===Arguments for generation===
    parser.add_argument("--gen", action="store_true", help="Only generate results")
    parser.add_argument("--start_idx", type=int, default=0, help="Start index for generation")
    parser.add_argument("--end_idx", type=int, default=None, help="End index for generation. If None, will generate until the end of the dataset.")

    # ===Arguments for evaluation===
    parser.add_argument("--eval", action="store_true", help="Only evaluate results")

    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if args.prompt_method:
        args.log_path_json = os.path.join(
            "logs_prompt", args.prompt_method, args.dataset, args.model, "generation_"+str(args.quantization)+".jsonl"
        )
        args.result_path_json = os.path.join(
            "results_prompt", args.prompt_method, args.dataset, args.model,
            f"{str(args.quantization)}_{args.start_idx}_{args.end_idx}_{timestamp}.json"
        )
    elif args.few_shot:
        args.log_path_json = os.path.join(
            "logs", "few_shot", args.dataset, args.model, "generation_"+str(args.quantization)+"_fewshot_"+str(args.num_examples)+".jsonl"
        )
        args.result_path_json = os.path.join(
            "results", "few_shot", args.dataset, args.model,
            f"{str(args.quantization)}_fewshot_{str(args.num_examples)}_{args.start_idx}_{args.end_idx}_{timestamp}.json"
        )
    else:
        args.log_path_json = os.path.join(
            "logs", args.dataset, args.model, "generation_"+str(args.quantization)+".jsonl"
        )
        args.result_path_json = os.path.join(
            "results", args.dataset, args.model,
            f"{str(args.quantization)}_{args.start_idx}_{args.end_idx}_{timestamp}.json"
        )

    os.makedirs(os.path.dirname(args.log_path_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.result_path_json), exist_ok=True)

    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Program starts")
    args = get_args()
    logging.info("Command line arguments:")
    for param, value in vars(args).items():
        logging.info("Argument %s: %s", param, value)
    
    model2id = {
        #general models
        "Llama-3.1-8B":("meta-llama/Llama-3.1-8B-Instruct", LlamaModel),
        "Llama-3.2-3B": ("meta-llama/Llama-3.2-3B-Instruct", LlamaModel),
        "Llama-3.2-1B": ("meta-llama/Llama-3.2-1B-Instruct", LlamaModel),
        "Llama-3.3-70B": ("meta-llama/Llama-3.3-70B-Instruct", LlamaModel),
        "Qwen2.5-3B": ("Qwen/Qwen2.5-3B-Instruct", QwenModel),
        "Qwen2.5-7B": ("Qwen/Qwen2.5-7B-Instruct", QwenModel),
        "Qwen2.5-14B": ("Qwen/Qwen2.5-14B-Instruct", QwenModel),
        "Qwen2.5-32B": ("Qwen/Qwen2.5-32B-Instruct", QwenModel),
        "Qwen2.5-72B": ("Qwen/Qwen2.5-72B-Instruct", QwenModel),
        "Qwen3-32B": ("Qwen/Qwen3-32B", QwenModel),
        "QwQ-32B": ("Qwen/QwQ-32B", QwenModel),
        "Gemma-2-27b": ("google/gemma-2-27b", GemmaModel),
        "Gemma-2-9b": ("google/gemma-2-9b-it", GemmaModel),
        "Mistral-3.2-24B": "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        "Phi-4": ("microsoft/phi-4", LlamaModel),
        "Deepseek-llm-65B": ("deepseek-ai/deepseek-llm-67b-chat", DeepseekModel),
        "gpt-oss-20b": ("openai/gpt-oss-20b", GptossModel),
        "gpt-oss-120b": ("openai/gpt-oss-120b", GptossModel),
        "gpt5": ("gpt-5", GPT),

        #medical models
        "Meditron-70B": ("epfl-llm/meditron-70b",ModelBase),
        "HuatuoGPT-o1-70B": ("FreedomIntelligence/HuatuoGPT-o1-70B", ModelBase),
        "ClinicalCamel-70B": ("wanglab/ClinicalCamel-70B", ModelBase),
        "PMC-LLaMA-13B": ("axiong/PMC_LLaMA_13B", ModelBase),
        "MedGemma-27B": ("google/medgemma-27b-text-it", GemmaModel),
        "Llama3-Med42-70B": ("m42-health/Llama3-Med42-70B", LlamaModel),
    }

    dataset2id = {
        "hoc": "MLC",
        "litcovid": "MLC",
        "bc5cdr": "NER",
        "ncbi": "NER",
        "chemprot": "RE",
        "ddi": "RE",
        "medqa": "QA",
        "pubmedqa": "QA",
        "cochranepls": "Simplification",
        "plos": "Simplification",
        "ms2": "Summarization",
        "pubmed": "Summarization",
    }

    prompt2class = {
        "cot": CoT,
        "sc_cot": CoT_SC,
        # "tot": ToT,
        # "self_refine": Self_Refine,
    }

    if args.model not in model2id:
        raise ValueError(f"Model {args.model} not supported. Available datasets: {list(model2id.keys())}")
    if args.dataset not in dataset2id:
        raise ValueError(f"Dataset {args.dataset} not supported. Available datasets: {list(dataset2id.keys())}")
    
    args.model_id = model2id[args.model][0]
    logging.info(f"MODEL: {args.model} ({args.model_id})")
    args.model_class = model2id[args.model][1]

    cls = getattr(dataset_manager, dataset2id[args.dataset.lower()])
    dataset = cls(args)
    args.end_idx = args.end_idx if args.end_idx is not None else len(dataset.data['test'])

    if args.prompt_method is None:
        args.prompt_class = PromptBaseMethod
    elif args.prompt_method in prompt2class:
        args.prompt_class = prompt2class[args.prompt_method]
    else:
        raise ValueError(f"Prompt method {args.prompt_method} not supported. Available prompt methods: {list(prompt2class.keys())}")

    if args.gen:
        generate(args, dataset)
    
    if args.eval:
        evaluate(args, dataset)

    logging.info("Program ends.")