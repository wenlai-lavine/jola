import os, json, random
import pandas as pd
from datasets import Dataset


class JoLADataset:
    def __init__(self, from_list=False, data_path=None, train_size=200, train_list=None):
        self.from_list = from_list
        self.train_list = train_list
        self.data_path = data_path
        self.train_size = train_size
        self.jola_datasets = {'train':{},'valid':{},'test':{}}


    def data_from_list(self):
        """Make dataset and collator for supervised fine-tuning."""
        raw_data = self.train_list
        # Only sample part of the training data if train_size !=0
        if self.train_size!=0:
            if len(raw_data) >= self.train_size:
                raw_data = random.sample(raw_data, self.train_size)
        formatted_data = self.format_prompt(raw_data, append_label=True)
        self.jola_datasets["train"] = formatted_data
        return self.jola_datasets
    

    def data_from_file(self):
        for split in ["train", "valid", "test"]:
            data_dir = os.path.join(self.data_path, f'{split}.json')
            raw_data = []
            with open(data_dir, 'r') as f:
                for line in f.readlines():
                    raw_data.append(json.loads(line.strip()))
            ### Only sample part of the training data if train_size !=0
            if self.train_size!=0 and split=='train':
                if len(raw_data) >= self.train_size:
                    raw_data = random.sample(raw_data, self.train_size)
            if self.train_size!=0 and split=='valid':
                if len(raw_data) >= self.train_size:
                    raw_data = random.sample(raw_data, self.train_size)
            if split == "test":
                formatted_data = self.format_prompt(raw_data, append_label=False)
            else:
                formatted_data = self.format_prompt(raw_data, append_label=True)
            self.jola_datasets[split] = formatted_data
        return self.jola_datasets
    
    def generate_prompt(self, instruction, input=None):
        if input:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                    ### Instruction: {instruction}
                    ### Input: {input}
                    ### Response:
                    """  # noqa: E501
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n ### Instruction:\n{instruction}\n\n### Response:\n"""  # noqa: E501
    

    def format_prompt(self, data_json_list, append_label):
        data = []
        for i in range(len(data_json_list)):
            instruction = data_json_list[i]["instruction"]
            answer = data_json_list[i]["answer"]

            if append_label:
                text = self.generate_prompt(instruction=instruction) + answer
                data.append({'text': text})
            else:
                text = self.generate_prompt(instruction=instruction)
                data.append({'prompt': text,'target_text': answer})
            
        df = pd.DataFrame(data=data)
        df = df.astype('str')
        data = Dataset.from_pandas(df)
        return data
    