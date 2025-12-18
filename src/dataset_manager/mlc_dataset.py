import os
import pandas as pd
import numpy as np
import json
import logging
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MLC:
    def __init__(self, args, dir=None):
        self.args = args

        current_dir = os.path.dirname(__file__)
        if self.args.dataset == 'hoc':
            self.dir = os.path.join(current_dir, '..', 'data/[MLC]Hoc/datasets/full_set')
        elif self.args.dataset == 'litcovid':
            self.dir = os.path.join(current_dir, '..', 'data/[MLC]LitCovid/datasets/full_set')
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
                'label': row['labels']
            }
            data.append(item)
        data = pd.DataFrame(data)
        return data
    
    def evaluate(self, response_groundtruth_pairs):
        """Evaluate the model predictions against the ground truth."""

        # Prepare all ground truths and predictions
        ground_truths = [item['label'].split(";") for item in response_groundtruth_pairs]
        predictions = [[it.strip().rstrip('.') for it in item['response'].split(";")] for item in response_groundtruth_pairs]

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
        if self.args.few_shot:
            examples = self.get_examples()
            prompt = (
                "### Instructions: \n"
                "You are an expert in multi-label classification. "
                "Given the input, classify it into one or more categories. "
                "The categories are: (" + ';'.join(self.get_categories()) + "). "
                "Your response should only be a semicolon-separated list of categories, without any other words or explanations.\n\n"
                "### Examples:\n" + "\n".join(self.examples) + "\n\n"
                "### Input: {input}\n\n"
                "### Response: "
            )
            return prompt
        
        if self.args.prompt_method:
            if self.args.prompt_method in ["cot", "sc_cot"]:
                return (
                    "### Instructions:\n"
                    "You are an expert in multi-label classification.\n"
                    "Analyze the input text and assign it to one or more appropriate categories.\n"
                    "Available categories: (" + '; '.join(self.get_categories()) + ").\n\n"
                    "### Input:\n{input}\n\n"
                    "First, briefly explain your reasoning step by step.\n"
                    "Then, provide the final classification result.\n"
                    "The result must be enclosed in a single pair of square brackets, with multiple categories separated by semicolons. For example: [aaa; bbb]\n"
                )
            else:
                raise ValueError(f"Unsupported prompt method: {self.args.prompt_method}")

        prompt = (
            "### Instructions: \n"
            "You are an expert in muliti-label classification. "
            "Given the input, classify it into one or more categories. "
            "The categories are: (" + ';'.join(self.get_categories()) + "). "
            "Your response should only be a semicolon-separated list of categories, without any other words or explainations.\n\n"
            "### Input: {input}\n\n"
            "### Response: "
        )
        return prompt
    
    def get_categories(self):
        """Return the unique categories from the dataset."""
        all_labels = set()
        for split in self.data:
            all_labels.update(self.data[split]['label'].str.split(';').explode().unique())
        return sorted(all_labels)
    
    def get_examples(self):
        """Return a list of examples from the dataset."""
        examples = []
        # Get examples from training data up to num_examples
        for _, data in self.data['train'].iloc[:self.args.num_examples].iterrows():
            example = f"Input: {data['text']}\nResponse: {data['label']}"
            examples.append(example)
        return examples

# Example usage:
if __name__ == "__main__":
    args = type('Args', (object,), {'dataset': 'hoc', "few_shot": True, "num_examples":1})  # Simulating args
    mlc = MLC(args)
    print(mlc.get_prompt())
    # print(mlc.get_examples())
    pass
