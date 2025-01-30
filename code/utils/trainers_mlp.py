import os
from trl import SFTTrainer
from utils.evaluate import evaluate_common_reason, evaluate_mmlu_pro, evaluate_gem
        
        
class CustomSFTTrainer(SFTTrainer):
    def __init__(self, model, train_dataset, eval_dataset, dataset_text_field, tokenizer, max_seq_length, data_collator, args, peft_config, callbacks):
        if callbacks:
            super().__init__(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                dataset_text_field=dataset_text_field,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                data_collator=data_collator,
                args=args,
                peft_config=peft_config,
                callbacks=callbacks,
            )
        else:
            super().__init__(
                model=model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                dataset_text_field=dataset_text_field,
                tokenizer=tokenizer,
                max_seq_length=max_seq_length,
                data_collator=data_collator,
                args=args,
                peft_config=peft_config,
            )

    def compute_loss(self, model, inputs,return_outputs=False):
        labels = inputs['labels']

        outputs = model(**inputs)
        
        ### Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        ### We don't use .loss here since the model may return tuples instead of ModelOutput.
        cn_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        loss = cn_loss
        ### Add L1 regularization term
        l1norm = 0
        l1_lambda=self.l1_lambda
        
        for param in model.parameters():
            if param.requires_grad:
                l1norm+=param.abs().sum()
        loss+=l1_lambda*l1norm
        if return_outputs:
            return loss,outputs
        else:
            return loss
       
    def test(self, fname, task, subtask, eval_dataset=None, model_name=None):
        self.model.eval()
        self.args.prediction_loss_only = False
        self.tokenizer.add_eos_token = False
        if not os.path.exists(fname):
            os.makedirs(fname)
        if task == "commonsense":
            evaluate_common_reason(eval_dataset=eval_dataset, task=task, subtask=subtask, model_name=model_name, model=self.model, tokenizer=self.tokenizer, fname=fname)
        elif task == "mmlu_pro":
            evaluate_mmlu_pro(eval_dataset=eval_dataset, task=task, subtask=subtask, model_name=model_name, model=self.model, tokenizer=self.tokenizer, fname=fname)
        elif task == "gem":
            evaluate_gem(eval_dataset=eval_dataset, task=task, subtask=subtask, model_name=model_name, model=self.model, tokenizer=self.tokenizer, fname=fname)
        else:
            print(f"please check the task name {task}")