import os
import pandas as pd
import numpy as np
import json
import logging
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class RE:
    def __init__(self, args, dir=None):
        self.args = args

        current_dir = os.path.dirname(__file__)

        if self.args.dataset == 'chemprot':
            self.dir = os.path.join(current_dir, '..', 'data/[RE]Chemprot/datasets/full_set')
        elif self.args.dataset == 'ddi':
            self.dir = os.path.join(current_dir, '..', 'data/[RE]DDI/datasets/full_set')
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
                'text': row['sentence'],
                'label': row['label']
            }
            data.append(item)
        data = pd.DataFrame(data)
        return data
    
    def evaluate(self, response_groundtruth_pairs):
        """Evaluate the model predictions against the ground truth."""

        # Prepare all ground truths and predictions
        ground_truths = [item['label'].strip() for item in response_groundtruth_pairs]
        predictions = [item['response'].strip() for item in response_groundtruth_pairs]

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
        if self.args.dataset == 'chemprot':
            prompt = (
                "### Instructions: \n"
                "You are an expert in chemical-protein relation classification. \n"
                "INPUT: the input is a sentence where the chemical is labeled as @CHEMICAL$ and the gene is labeled as @GENE$ accordingly in a sentence. \n"
                "OUTPUT: your task is to select one out of the six types of relations ('CPR:3', 'CPR:4', 'CPR:5', 'CPR:6', 'CPR:9', and 'false') for the gene and chemical. You must output ONLY one of the following labels. DO NOT output anything else. No explanations, no punctuation, no quotation marks, no newline, just the exact label. \n\n"
                "CPR:3, which includes UPREGULATOR, ACTIVATOR, and INDIRECT UPREGULATOR \n"
                "CPR:4, which includes DOWNREGULATOR, INHIBITOR ,and INDIRECT DOWNREGULATOR \n"
                "CPR:5, which includes AGONIST, AGONIST ACTIVATOR, and AGONIST INHIBITOR \n"
                "CPR:6, which includes ANTAGONIST \n"
                "CPR:9, which includes SUBSTRATE, PRODUCT OF and SUBSTRATE PRODUCT OF \n"
                "false, which indicates no relations\n\n"
            )
        elif self.args.dataset == 'ddi':
            prompt = (
                "### Instructions: \n"
                "You are an expert in drug-drug relation classification. \n"
                "INPUT: the input is a sentence where the drugs are labeled as @DRUG$. \n"
                "OUTPUT: your task is to select one out of the five types of relations ('DDI-effect', 'DDI-mechanism', 'DDI-advise', 'DDI-false', and 'DDI-int') for the drugs. You must output ONLY one of the following labels. DO NOT output anything else. No explanations, no punctuation, no quotation marks, no newline, just the exact label.\n\n"
                "DDI-mechanism: This type is used to annotate DDIs that are described by their PK mechanism (e.g. Grepafloxacin may inhibit the metabolism of theobromine)\n"
                "DDI-effect: This type is used to annotate DDIs describing an effect (e.g. In uninfected volunteers, 46% developed rash while receiving SUSTIVA and clarithromycin) or a PD mechanism (e.g. Chlorthalidone may potentiate the action of other antihypertensive drugs)\n"
                "DDI-advise: This type is used when a recommendation or advice regarding a drug interaction is given (e.g. UROXATRAL should not be used in combination with other alpha-blockers)\n"
                "DDI-int: This type is used when a DDI appears in the text without providing any additional information (e.g. The interaction of omeprazole and ketoconazole has been established)\n"
                "DDI-false, This type is used when no DDI relation appears\n\n"
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
        grouped = self.data['train'].groupby('label')
        
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
    args = type('Args', (object,), {'dataset': 'ddi', "few_shot": True, "num_examples":7})  # Simulating args
    re = RE(args)
    print(re.get_prompt())
    # print(mlc.get_examples())
    pass