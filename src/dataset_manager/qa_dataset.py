import os
import pandas as pd
import numpy as np
import json
import logging
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class QA:
    def __init__(self, args, dir=None):
        self.args = args

        current_dir = os.path.dirname(__file__)
        if self.args.dataset == 'medqa':
            self.dir = os.path.join(current_dir, '..', 'data/[QA]MedQA/datasets/full_set')
        elif self.args.dataset == 'pubmedqa':
            self.dir = os.path.join(current_dir, '..', 'data/[QA]PubMedQA/datasets/full_set')
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")

        self.data = None
        # Paths for train, dev, and test sets
        self.train_file = os.path.join(self.dir, 'train.tsv')
        self.dev_file = os.path.join(self.dir, 'dev.tsv')
        self.test_file = os.path.join(self.dir, 'test.tsv')

        self.read_files()

        logging.info("=====dataset examples=====")
        # print(self.data)
        print(self.data['test'][0:5])
        print(self.data['test'].iloc[0].to_dict())
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
        if self.args.dataset == 'medqa':
            try:
                df = pd.read_csv(file_path, sep='\t', header=0)
                # return df
                data = []
                for _, row in df.iterrows():
                    options = eval(row['options'])
                    item = {
                        'text': row['question']+"\n" + "\n".join([opt['key'] + ". " + opt['value'] for opt in options]),
                        'label': row['answer_idx']
                    }
                    data.append(item)
                data = pd.DataFrame(data)
                return data
            except Exception as e:
                return None
        elif self.args.dataset == 'pubmedqa':
            try:
                df = pd.read_csv(file_path, sep='\t', header=0)
                data = []
                for _, row in df.iterrows():
                    item = {
                        'text': row['QUESTION'] +"\n" + "Abstract: " + " ".join(eval(row['CONTEXTS'])),
                        'label': row['final_decision']
                    }
                    data.append(item)
                data = pd.DataFrame(data)
                return data
            except Exception as e:
                return None
        else:
            raise ValueError(f"Unsupported dataset: {self.args.dataset}")
    
    def evaluate(self, response_groundtruth_pairs):
        """Evaluate the model predictions against the ground truth."""

        # Prepare all ground truths and predictions
        ground_truths = [[item['label'].strip()] for item in response_groundtruth_pairs]
        predictions = [[item['response'].split(".")[0].strip()] for item in response_groundtruth_pairs]

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
            "scores": accuracy
        }

        
        print(json.dumps(metrics, indent=4))
        return metrics
    
    def get_prompt(self):
        """Return the prompt format for the dataset."""

        if self.args.prompt_method:
            if self.args.prompt_method in ["cot", "sc_cot"]:
                if self.args.dataset == 'medqa':
                    return (
                        "### Instructions:\n"
                        "Your task is to answer medical questions with the provided choices.\n\n"
                        "### Input:\n{input}\n\n"
                        "First, briefly explain your reasoning step by step.\n"
                        "Then, output only the answer option ([A], [B], [C], [D], or [E]) enclosed in square brackets.\n"
                        "Example: the final answer is [C]\n\n"
                    )
                elif self.args.dataset == 'pubmedqa':
                    return (
                        "### Instructions:\n"
                        "Your task is to answer biomedical questions using the given abstract.\n\n"
                        "### Input:\n{input}\n\n"
                        "First, briefly explain your reasoning step by step.\n"
                        "Then, output only one of the following options: [yes], [no], or [maybe].\n\n"
                    )
                else:
                    raise ValueError(f"Unsupported dataset for CoT prompt: {self.args.dataset}")
            else:
                raise ValueError(f"Unsupported prompt method: {self.args.prompt_method}")


        if self.args.dataset == 'medqa':
            prompt = (
                "### Instructions: \n"
                "Your task is to answer medical questions with the provided choices. Only output the answer option (A/B/C/D/E) as answer. \n"
                "INPUT: The input consists of a question followed by several choices. \n"
                "OUTPUT: Answer each question by providing one of the following options: A, B, C, D, E. \n\n"
            )
        elif self.args.dataset == 'pubmedqa':
            prompt = (
                "### Instructions: \n"
                "Your task is to answer biomedical questions using the given abstract. Only output yes, no, or maybe as answer.  \n"
                "INPUT: The input is a question followed by an abstract. \n"
                "OUTPUT: Answer each question by providing one of the following options: yes, no, maybe. \n\n"
            )

        if self.args.few_shot:
            examples = self.get_examples()
            prompt += "### Example(s):\n" + "\n".join(examples) + "\n\n"

        prompt += (
            "### Input: {input}\n\n"
            "### Output: "
        )

        
        return prompt
    
    def get_examples(self):
        """Return a list of examples from the dataset with balanced labels."""
        examples = []
        # Group training data by label
        try:
            grouped = self.data['train'].groupby('label')
        except:
            grouped = self.data['test'].groupby('label')
        
        # Calculate examples needed per label
        num_labels = len(grouped)
        if self.args.num_examples % num_labels == 0:
            examples_per_label = self.args.num_examples // num_labels
        else:
            examples_per_label = self.args.num_examples // num_labels + 1 
        
        # Get balanced examples from each label type
        for label, group in grouped:
            # Get examples_per_label samples from this group
            for i in range(examples_per_label):
                example = f"Input: {group.iloc[i]['text']}\nOutput: {label}"
                examples.append(example)
                    
        return examples[:self.args.num_examples]

# Example usage:
if __name__ == "__main__":
    args = type('Args', (object,), {'dataset': 'medqa', "few_shot": True, "num_examples":7})  # Simulating args
    qa = QA(args)
    print(qa.get_prompt())
    # print(mlc.get_examples())
    pass
    