import os
import pandas as pd
import numpy as np
import json
import logging
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from collections import defaultdict

class Summarization:
    def __init__(self, args, dir=None):
        self.args = args

        self.data = {}
        self.load_data()

        logging.info("=====dataset examples=====")
        print(self.data['test'][0:5])
        # print(self.data['test'].iloc[0].to_dict())
        logging.info("==============================")

    def load_data(self):
        if self.args.dataset == 'ms2':
            self.data['test'] = pd.DataFrame(load_dataset("allenai/mslr2022", "ms2", split='validation', streaming=True))
            self.data['test'] = self.data['test'][['abstract', 'target']]
            def apply_ms2(row):
                row['abstract'] = ' '.join(row['abstract'])
                return row
            self.data['test'] = self.data['test'].apply(apply_ms2, axis=1)
            self.data['test'].rename(columns={'abstract': 'text', 'target': 'label'}, inplace=True)
        elif self.args.dataset == 'pubmed':
            self.data['test'] = pd.DataFrame(load_dataset("ccdv/pubmed-summarization", "section", split='test', streaming=True))
            self.data['test'] = self.data['test'][['article', 'abstract']]
            self.data['test'].rename(columns={'article': 'text', 'abstract': 'label'}, inplace=True)
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")
    

        
    # def _read_tsv(self, file_path):
    #     """Read a single TSV file and return structured data."""
    #     try:
    #         df = pd.read_csv(file_path, sep='\t', header=0)
    #         data = []
    #         for _, row in df.iterrows():
    #             item = {
    #                 'text': row['source'],
    #                 'label': row['target']
    #             }
    #             data.append(item)
    #         data = pd.DataFrame(data)
    #         return data
    #     except Exception as e:
    #         logging.error(f"Error reading file {file_path}: {e}")
    #         return None
    
    def evaluate(self, response_groundtruth_pairs):
        """Evaluate the similarity between predicted and ground truth sentences."""

        metrics = defaultdict(list)
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        for item in response_groundtruth_pairs:
            ground_truth = item['label'].strip()
            prediction = item['response'].strip()

            # Calculate ROUGE scores
            rouge_scores = scorer.score(ground_truth, prediction)
            metrics['rouge1'].append(rouge_scores['rouge1'].fmeasure)
            metrics['rouge2'].append(rouge_scores['rouge2'].fmeasure)
            metrics['rougeL'].append(rouge_scores['rougeL'].fmeasure)

            # Calculate BLEU score
            try:
                bleu = sentence_bleu([ground_truth.split()], prediction.split())
                metrics['bleu'].append(bleu)
            except:
                metrics['bleu'].append(0.0)

        # Calculate averages
        final_metrics = {
            'rouge1': float(np.mean(metrics['rouge1'])),
            'rouge2': float(np.mean(metrics['rouge2'])),
            'rougeL': float(np.mean(metrics['rougeL'])),
            'bleu': float(np.mean(metrics['bleu'])),
            'scores': float(np.mean(metrics['rougeL']))  # Using ROUGE-L as main score
        }

        print(json.dumps(final_metrics, indent=4))
        return final_metrics
    
    def get_prompt(self):
        """Return the prompt format for the dataset."""
        prompt = (
            "### Instructions: \n"
            "The task is to summarize an input biomedical literature in six sentences. \n"
            "INPUT: the input is a biomedical literature. \n"
            "OUTPUT: the output is the summary of an input biomedical literature in six sentences. \n\n"
            "### Input: {input}\n\n"
            "### Output: "
        )
        return prompt
    

# Example usage:
if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.dataset = 'pubmed'  # or 'pubmed' 'ms2'
    
    args = Args()
    ms2_dataset = Summarization(args)

