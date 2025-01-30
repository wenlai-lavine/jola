import os
from transformers import AutoTokenizer, TrainingArguments, set_seed, logging, EarlyStoppingCallback
from models.modeling_llama import LlamaForCausalLM
from models.modeling_qwen2 import Qwen2ForCausalLM
import torch
import torch.nn as nn
from trl import DataCollatorForCompletionOnlyLM
import argparse
import numpy as np
import random
from utils.dataloaders import COMMON_REASON, MMLU_PRO, GEM
from utils.trainers import CustomSFT_Trainer

parser = argparse.ArgumentParser()
parser.add_argument('--lr',type=float,default=1e-4)
parser.add_argument('--train_batch',type=int,default=8)
parser.add_argument('--num_epoch',type=int,default=10)
parser.add_argument('--train_size',type=int,default=0)
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--output_dir', type=str,default=None)
parser.add_argument('--eval_batch',type=int,default=8)
parser.add_argument('--task', type=str,help='The task dataset to train on')
parser.add_argument('--output_file_name',type=str,help='The name of the output file')
parser.add_argument('--applied_module',type=str,default='attention',help='The modules to apply lofit; attention by default')
parser.add_argument('--base_model_name',type=str,default='llama3-8b-instruct',help='The model base to train on',required=True)
parser.add_argument('--hf_cache_dir',type=str,default='Cache Path',required=False,help='The cache directory for huggingface models')
parser.add_argument('--device',type=str,default='cuda',required=False,help='The device to load the model; cuda by default')
parser.add_argument('--save_strategy',type=str,default='best',required=False,help='The strategy to save the model: best: only save the best model; no: do not save the model')
parser.add_argument('--subtask', type=str, default='',required=False)
parser.add_argument('--data_path', type=str, default='',required=False)
parser.add_argument('--gated_lambda', type=float, default=0, help='group lasso regularization lambda for lofit',required=False)
parser.add_argument('--gate_scheduler', type=str, help='group lasso regularization lambda for lofit',required=False, default="")


args = parser.parse_args()
### Turn Wandb log on if it is in train mode
wandb.init(mode="online",name=args.output_dir.split("/")[-1])

### Load training hyperparametres
lr = float(args.lr)
train_batch_size = int(args.train_batch)
eval_batch_size = int(args.eval_batch)
num_epoch = int(args.num_epoch)
applied_module = args.applied_module
output_dir = args.output_dir
device = args.device


## Set all random seeds for reproducibility
seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
set_seed(seed)

logging.set_verbosity_error()
### Maps of model names and task names
### If you want to use your own model, please add the model name to the map
models_map = {
    'llama3.1_8B': 'meta-llama/Meta-Llama-3.1-8B',
    'llama3.1_8B_chat': 'meta-llama/Llama-3.1-8B-Instruct',
    'llama3.2_1B_chat': 'meta-llama/Llama-3.2-1B-Instruct',
    'llama3.2_3B_chat': 'meta-llama/Llama-3.2-3B-Instruct',
    'llama3.1_70B_chat': 'meta-llama/Llama-3.1-70B-Instruct',
    'qwen2.5_7B_chat': 'Qwen/Qwen2.5-7B-Instruct',
}

task_map = {
    "commonsense": {
        "dataloader": COMMON_REASON(args.data_path, args.base_model_name, args.task, args.subtask),
        "trainer": CustomSFT_Trainer
    },
    "mmlu_pro": {
        "dataloader": MMLU_PRO(args.data_path, args.base_model_name, args.task, args.subtask),
        "trainer": CustomSFT_Trainer
    },
    "gem": {
        "dataloader": GEM(args.data_path, args.base_model_name, args.task, args.subtask),
        "trainer": CustomSFT_Trainer
    }
}

if not args.base_model_name in models_map:
    raise ValueError(f'The base model {args.base_model_name} is not supported')

### Load tokenizers and models
model_name = models_map[args.base_model_name]
cache_dir = args.hf_cache_dir
tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir=cache_dir)

### Use right padding for training
tokenizer.padding_side = 'right'    
if '13b' in model_name or 'llama' in model_name:
    ## Use bfloat16 training for 13B models and Gemma
    torch_dtype = torch.bfloat16
    bf16 = True
else:
    torch_dtype = torch.float32
    bf16 = False

peft_config = None
if 'llama' in model_name:
    model = LlamaForCausalLM.custom_from_pretrained(model_name,
                                            device_map=device, 
                                            cache_dir=cache_dir,
                                            applied_module = applied_module,
                                            torch_dtype=torch_dtype)
elif 'Qwen' in model_name:
    model = Qwen2ForCausalLM.custom_from_pretrained(model_name,
                                            device_map=device, 
                                            cache_dir=cache_dir,
                                            applied_module = applied_module,
                                            torch_dtype=torch_dtype)
else:
    raise ValueError(f'{model_name} is not supported!')

model.model.train()
### Define padding
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(model.config.vocab_size + 1)

count = 0

### First freeze all pretrained parameters
for param in model.parameters():
    param.requires_grad = False
trainable_params = []
num_params = 0
### Unfreeze editing moudules [attention] for training
for i in range(model.config.num_hidden_layers):
    if applied_module == 'attention':
        attn_A = model.model.layers[i].self_attn.attn_A
        for j,module in enumerate(attn_A):
            trainable_params.append(module)
            module.requires_grad = True
            num_params+=module.numel()
        attn_v = model.model.layers[i].self_attn.attn_v
        for j,module in enumerate(attn_v):
            trainable_params.append(module)
            module.requires_grad = True
            num_params+=module.numel()
        g1 = model.model.layers[i].self_attn.log_g1
        g1.requires_grad = True
        g2 = model.model.layers[i].self_attn.log_g2
        g2.requires_grad = True
print('trainable params:',num_params)

if args.save_strategy == 'best':
    save_strategy = 'epoch'
    load_best_model_at_end = True
    save_total_limit = 1
elif args.save_strategy == "steps":
    save_strategy = "steps",
    evaluation_strategy="steps",
    eval_steps=5
elif args.save_strategy == 'no':
    save_strategy = 'no'
    load_best_model_at_end = False
    save_total_limit = None
else:
    raise ValueError(f'Save strategy {args.save_strategy} is not supported')

# early stop according to the performance from validation set
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=2, # 如果验证集性能在3个评估周期内没有提升，则停止训练 
    early_stopping_threshold=0.0 # 性能提升的最小阈值
)

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=lr,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    num_train_epochs=num_epoch,
    evaluation_strategy="epoch",
    save_strategy=save_strategy,
    load_best_model_at_end=load_best_model_at_end,
    save_total_limit = save_total_limit,
    report_to='wandb',
    logging_strategy='epoch',
    seed = seed,
    do_train = True,
    do_eval = True,
    bf16=bf16
)

torch.autograd.set_detect_anomaly(True)

datasets = task_map[args.task]['dataloader'].load_data(train_size=args.train_size)

for key in ['train','valid','test']:
    print(f"Number of {key} samples: {len(datasets[key])}")

trainer = task_map[args.task]['trainer']

## set to only tuned new tokens in the input_ids
response_template_with_context = "### Response:\n"

data_collator = DataCollatorForCompletionOnlyLM(response_template=response_template_with_context, tokenizer=tokenizer, mlm=False)

trainer = trainer(
    model,
    train_dataset=datasets['train'],
    eval_dataset = datasets['valid'],
    dataset_text_field = 'text',
    tokenizer=tokenizer,
    max_seq_length=400,
    data_collator = data_collator,
    args=training_args,
    peft_config = peft_config,
    callbacks=[early_stopping_callback],
    gate_scheduler=args.gate_scheduler
)

if not args.gate_scheduler:
    trainer.gated_lambda = args.gated_lambda
for i in range(model.config.num_hidden_layers):
    if applied_module == 'attention':
        attn_A = model.model.layers[i].self_attn.attn_A
        for j,module in enumerate(attn_A):
            nn.init.normal_(module,mean=0,std=1e-3)
        attn_v = model.model.layers[i].self_attn.attn_v
        for j,module in enumerate(attn_v):
            nn.init.normal_(module,mean=0,std=1e-3)
        
        g1 = model.model.layers[i].self_attn.log_g1
        nn.init.xavier_uniform_(g1)
        g2 = model.model.layers[i].self_attn.log_g2
        nn.init.xavier_uniform_(g2)
trainer.train()

## do evaluation
trainer.test(fname=args.output_file_name, task=args.task, subtask=args.subtask, eval_dataset = datasets['test'],model_name = args.base_model_name)