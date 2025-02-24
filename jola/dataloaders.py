import os
import json
import pandas as pd
from datasets import Dataset
import random

###
class COMMON_REASON:
    ## instuction/answer
    def __init__(self, data_path=None, model_name=None, task_name=None, subtask_name=None):
        self.data_path = data_path
        self.task_name = task_name
        self.subtask_name = subtask_name
        self.model_name = model_name
        self.datasets = {'train':{},'valid':{},'test':{}}
        self.prompt_template = "Instruction: {} Predict Answer: "
    
    def generate_prompt(self, instruction, input=None):
        if input:
            return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                    ### Instruction: {instruction}
                    ### Input: {input}
                    ### Response:
                    """  # noqa: E501
        else:
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.\n ### Instruction:\n{instruction}\n\n### Response:\n"""  # noqa: E501

    def load_data(self, train_size=0):
        for split in self.datasets.keys():
            data_dir = os.path.join(self.data_path, self.task_name, self.subtask_name, f'{split}.json')
            raw_data = []
            with open(data_dir, 'r') as f:
                for line in f.readlines():
                    raw_data.append(json.loads(line.strip()))
            ### Only sample part of the training data if train_size !=0
            if train_size!=0 and split=='train':
                if len(raw_data) >= train_size:
                    raw_data = random.sample(raw_data, train_size)
            if train_size!=0 and split=='valid':
                if len(raw_data) >= train_size:
                    raw_data = random.sample(raw_data, train_size)
            if split == "test":
                formatted_data = self.format_prompt(raw_data, append_label=False)
            else:
                formatted_data = self.format_prompt(raw_data, append_label=True)
            self.datasets[split] = formatted_data
        return self.datasets


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
        data = Dataset.from_pandas(df)
        return data
    
### MMLU_PRO
class MMLU_PRO:
    ## 
    def __init__(self, data_path=None, model_name=None, task_name=None, subtask_name=None):
        self.data_path = data_path
        self.task_name = task_name
        self.subtask_name = subtask_name
        self.model_name = model_name
        self.datasets = {'train':{},'valid':{},'test':{}}

    def load_data(self, train_size=0):
        for split in self.datasets.keys():
            data_dir = os.path.join(self.data_path, self.task_name, self.subtask_name, f'{split}.json')
            raw_data = []
            with open(data_dir, 'r') as f:
                for line in f.readlines():
                    raw_data.append(json.loads(line.strip()))
            ### Only sample part of the training data if train_size !=0
            if train_size!=0 and split=='train':
                if train_size >= len(raw_data):
                    raw_data = random.sample(raw_data, train_size)
            if split == "test":
                formatted_data = self.format_prompt(raw_data, append_label=False)
            else:
                formatted_data = self.format_prompt(raw_data, append_label=True)
            self.datasets[split] = formatted_data
        return self.datasets
    
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


### GEM
class GEM:
    ## 
    def __init__(self, data_path=None, model_name=None, task_name=None, subtask_name=None):
        self.data_path = data_path
        self.task_name = task_name
        self.subtask_name = subtask_name
        self.model_name = model_name
        self.datasets = {'train':{},'valid':{},'test':{}}

    def load_data(self, train_size=0):
        for split in self.datasets.keys():
            data_dir = os.path.join(self.data_path, self.task_name, self.subtask_name, f'{split}.json')
            raw_data = []
            with open(data_dir, 'r') as f:
                for line in f.readlines():
                    raw_data.append(json.loads(line.strip()))
            ### Only sample part of the training data if train_size !=0
            if train_size!=0 and split=='train':
                if len(raw_data) >= train_size:
                    raw_data = random.sample(raw_data, train_size)
            if train_size!=0 and split=='valid':
                if len(raw_data) >= train_size:
                    raw_data = random.sample(raw_data, train_size)
            if split == "test":
                formatted_data = self.format_prompt(raw_data, append_label=False)
            else:
                formatted_data = self.format_prompt(raw_data, append_label=True)
            self.datasets[split] = formatted_data
        return self.datasets
    

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