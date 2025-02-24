from transformers import AutoTokenizer, TrainingArguments, EarlyStoppingCallback

import sys, os, torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from jola import JoLAConfig, JoLAModel, JoLATrainer, JoLADataset, make_data_collator

## if you have already install jola through pip, you can directly import them
# from jola import JoLAConfig, JoLAModel, JoLATrainer, JoLADataset, make_data_collator

# set the jola config through a yamal file, please use your own yamal by setting 'default=False' and specify a 'yaml' file
jola_config = JoLAConfig.get_jola_config(default=True)

jola_tokenizer = AutoTokenizer.from_pretrained(**jola_config["model_config"])

# Use right padding for training
jola_tokenizer.padding_side = 'right'

# Load models
jola_model = JoLAModel.jola_from_pretrained(**jola_config["model_config"])

# unfreeze jola parameters
jola_model.unfreeze_jola_params()

# set in training mode
jola_model.model.train()

# Define padding
if jola_tokenizer.pad_token is None:
    jola_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    jola_model.resize_token_embeddings(jola_model.config.vocab_size + 1)

# data setting, data loader
data_collator = make_data_collator(tokenizer=jola_tokenizer)

# dataset setting
jola_dataset = JoLADataset(data_path=jola_config["data_config"]["data_path"])
jola_data = jola_dataset.data_from_file()

# early stop according to the performance from validation set
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=2,
    early_stopping_threshold=0.0
)

training_args = TrainingArguments(**jola_config["training_config"])

# trainer
jola_trainer = JoLATrainer(
    jola_model,
    train_dataset=jola_data['train'],
    eval_dataset = jola_data['valid'],
    tokenizer=jola_tokenizer,
    data_collator = data_collator,
    args=training_args,
    callbacks=[early_stopping_callback],
    gate_scheduler=jola_config["jola_config"]["gate_scheduler"]
)

torch.autograd.set_detect_anomaly(True)

# set gate schedule
if not jola_config["jola_config"]["gate_scheduler"]:
    jola_trainer.gated_lambda = jola_config['training_config']["gate_lambda"]

jola_trainer.train()

# do evaluation ** double check
# jola_trainer.test(fname=args.output_file_name, task=args.task, subtask=args.subtask, eval_dataset=jola_data['test'],model_name = args.base_model_name)