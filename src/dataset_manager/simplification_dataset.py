import os
import pandas as pd
import numpy as np
import json
import logging
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from collections import defaultdict

class Simplification:
    def __init__(self, args, dir=None):
        self.args = args

        current_dir = os.path.dirname(__file__)

        if self.args.dataset == 'cochranepls':
            self.dir = os.path.join(current_dir, '..', 'data/[Simplification]CochranePLS/datasets/full_set')
        elif self.args.dataset == 'plos':
            self.dir = os.path.join(current_dir, '..', 'data/[Simplification]PLOS/datasets/full_set')
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        self.data = None
        # Paths for train, dev, and test sets
        self.train_file = os.path.join(self.dir, 'train.tsv')
        self.dev_file = os.path.join(self.dir, 'dev.tsv')
        self.test_file = os.path.join(self.dir, 'test.tsv')

        self.read_files()

        logging.info("=====dataset examples=====")
        print(self.data['test'][0:5])
        logging.info("==============================")

    def read_files(self):
        """Read all TSV files and store them in a structured format."""
        self.data = {
            'train': self._read_tsv(self.train_file),
            'dev': self._read_tsv(self.dev_file),
            'test': self._read_tsv(self.test_file)
        }
        
    def _read_tsv(self, file_path):
        """Read a single TSV file and return structured data."""
        try:
            df = pd.read_csv(file_path, sep='\t', header=0)
            data = []
            for _, row in df.iterrows():
                item = {
                    'text': row['source'],
                    'label': row['target']
                }
                data.append(item)
            data = pd.DataFrame(data)
            return data
        except Exception as e:
            logging.error(f"Error reading file {file_path}: {e}")
            return None
    
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
            "The task is to simplify the input abstract of a biomedical literature. \n"
            "INPUT: the input is the abstract of a biomedical literature. \n"
            "OUTPUT: the output is the simplified abstract for the input abstract of a biomedical literature. No extra explaination or analysis.\n\n"
            "### Input: {input}\n\n"
            "### Output: "
        )
        return prompt
    

# Example usage:
if __name__ == "__main__":
    # hoc = HocDataset()
    pass
