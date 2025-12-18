import os
import pandas as pd
import numpy as np
import json
import re
import logging
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class NER:
    def __init__(self, args, dir=None):
        self.args = args
        
        current_dir = os.path.dirname(__file__)
        if self.args.dataset == 'bc5cdr':
            self.dir = os.path.join(current_dir, '..', 'data/[NER]BC5CDR_Chemical/datasets/instruction')
        elif self.args.dataset == 'ncbi':
            self.dir = os.path.join(current_dir, '..', 'data/[NER]NCBI_Disease/datasets/instruction')
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

        self.examples = self.get_examples()

    def read_files(self):
        """Read all TSV files and store them in a structured format."""
        self.data = {
            'train': self._read_tsv(self.train_file),
            'dev': self._read_tsv(self.dev_file),
            'test': self._read_tsv(self.test_file)
        }
        
    def _read_tsv(self, file_path):
        """Read a single TSV file and return structured data."""
        df = pd.read_csv(file_path, sep='\t', header=0)
        data = []
        for _, row in df.iterrows():
            item = {
                'text': row['text'],
                'label': row['label']
            }
            data.append(item)
        data = pd.DataFrame(data)
        return data
    
    def evaluate(self, response_groundtruth_pairs):
        """Evaluate the model predictions against the ground truth."""

        # Prepare all ground truths and predictions
        ground_truths = [
            [a.strip() for a in re.findall(r'<span\s+class="[^"]*">(.*?)</span>', item['label'])] for item in response_groundtruth_pairs
        ]
        predictions = [
            [a.strip() for a in re.findall(r'<span\s+class="[^"]*">(.*?)</span>', item['response'])] for item in response_groundtruth_pairs
        ]

        # Transform using MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        y_true = mlb.fit_transform(ground_truths)
        y_pred = mlb.transform(predictions)

        # Calculate metrics for all samples at once
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            "scores": f1
        }

        
        print(json.dumps(metrics, indent=4))
        return metrics
    
    def get_prompt(self):
        """Return the prompt format for the dataset."""
        if self.args.dataset == 'bc5cdr':
            prompt = (
                "### Instructions: \n"
                "In the sentence extracted from biomedical literature, identify all the chemical entities. The required answer format is the same sentence with HTML <span> tags to mark up specific entities. "
                "Do not include any other words or explainations in response.\n\n"
                "### Entity Markup Guides:\n"
                'Use <span class="chemical"> to denote a chemical.\n\n'
            )
        elif self.args.dataset == 'ncbi':
            prompt = (
                "### Instructions: \n"
                "In the sentence extracted from biomedical literature, identify all the chemical entities. The required answer format is the same sentence with HTML <span> tags to mark up specific entities. "
                "Do not include any other words or explainations in response.\n\n"
                "### Entity Markup Guides:\n"
                'Use <span class="disease"> to denote a disease.\n\n'
            )

        if self.args.few_shot:
            examples = self.get_examples()
            prompt += "\n### Examples:\n" + "\n".join(examples) + "\n\n"

        prompt += (
            "### Input Text: {input}\n\n"
            "### Output Text: "
        )
        return prompt
    
    def get_examples(self):
        """Return a list of examples from the dataset."""
        examples = []
        # Get examples from training data up to num_examples
        for _, data in self.data['train'].iloc[:self.args.num_examples].iterrows():
            example = f"Input Text: {data['text']}\nOutput Text: {data['label']}\n"
            examples.append(example)
        return examples


# Example usage:
if __name__ == "__main__":
    args = type('Args', (object,), {'dataset': 'ncbi', "few_shot": True, "num_examples":5})  # Simulating args
    ner = NER(args)
    print(ner.get_prompt())
    # print(mlc.get_examples())
    pass
