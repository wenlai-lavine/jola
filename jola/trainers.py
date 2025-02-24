import os
import math, torch
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
# from utils.evaluate import evaluate_common_reason, evaluate_mmlu_pro, evaluate_gem

# define data collator for jola
def make_data_collator(response_template="### Response:\n", tokenizer=None, mlm=False):
    data_collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer, mlm=mlm)
    return data_collator


# define four schedule
class LinearSchedule:
    def __init__(self, start_lambda, end_lambda, total_steps):
        self.start_lambda = start_lambda
        self.end_lambda = end_lambda
        self.total_steps = total_steps
        self.step_count = 0

    def get_lambda(self):
        self.step_count += 1
        return self.start_lambda + (self.end_lambda - self.start_lambda) * (self.step_count / self.total_steps)


class CyclicSchedule:
    def __init__(self, cycle_length, total_steps):
        self.cycle_length = cycle_length
        self.total_steps = total_steps
        self.step_count = 0

    def get_lambda(self):
        self.step_count += 1
        return 0.5 + 0.5 * math.sin(2 * math.pi * (self.step_count / self.cycle_length))

class PerformanceBasedSchedule:
    def __init__(self, initial_lambda, adjustment_factor=0.01):
        self.current_lambda = initial_lambda
        self.adjustment_factor = adjustment_factor
        self.step_count = 0

    def get_lambda(self, performance_improvement):
        self.step_count += 1
        if performance_improvement < 0:  # If performance improves, increase λ
            self.current_lambda = min(1.0, self.current_lambda + self.adjustment_factor)
        else:  # If performance worsens, decrease λ
            self.current_lambda = max(0.0, self.current_lambda - self.adjustment_factor)
        return self.current_lambda

class ExponentialDecaySchedule:
    def __init__(self, start_lambda, decay_rate):
        self.start_lambda = start_lambda
        self.decay_rate = decay_rate
        self.step_count = 0

    def get_lambda(self):
        self.step_count += 1
        return self.start_lambda * math.exp(-self.decay_rate * self.step_count)


class JoLATrainer(SFTTrainer):
    def __init__(self, model, train_dataset, eval_dataset, tokenizer, data_collator, args, callbacks, gate_scheduler, dataset_text_field="text", max_seq_length=400, peft_config=None):
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
                # callbacks=callbacks,
            )
        self.gate_scheduler = gate_scheduler
        self.num_steps = (len(train_dataset) // args.per_device_train_batch_size) * args.num_train_epochs
        if self.gate_scheduler == "linear":
            self.lambda_scheduler = LinearSchedule(0.0, 0.2, self.num_steps)
            self.gated_lambda = 0.0
        elif self.gate_scheduler == "cyclic":
            self.lambda_scheduler = CyclicSchedule(cycle_length=20, total_steps=self.num_steps)
            self.gated_lambda = 0.1
        elif self.gate_scheduler == "perform":
            self.lambda_scheduler = PerformanceBasedSchedule(initial_lambda=0.1)
            self.gated_lambda = 0.1
        elif self.gate_scheduler == "expon":
            self.lambda_scheduler = ExponentialDecaySchedule(start_lambda=0.1, decay_rate=0.01)
            self.gated_lambda = 0.1
        
        ## g1 and g2 status during training
        self.g1_prop = []
        self.g2_prop = []
        ## save loss for last step
        self.last_loss = 50
    
    def get_penalty(self, log_alpha, stretch_limits=(-0.1, 1.1), temperature=0.33, eps=1e-6):
        low, high = torch.tensor(stretch_limits)
        assert low < 0.0, "p_gate_closed can be computed only if lower stretch limit is negative"
        p_open = torch.sigmoid(log_alpha - temperature * torch.log(-low / high))
        p_open = torch.clamp(p_open, eps, 1.0 - eps)
        total_reg = torch.sum(p_open)
        return total_reg / p_open.size(0)
    
    def get_gates(self, log_gate, is_train, stretch_limits=(-0.1, 1.1), temperature=0.33, eps=1e-6):
        """ samples gate activations in [0, 1] interval """
        low, high = stretch_limits
        if is_train:
            shape = log_gate.size()
            noise = (1 - 2*eps) * torch.rand(shape).to(log_gate.device) + eps
            concrete = torch.sigmoid((torch.log(noise) - torch.log(1 - noise) + log_gate) / temperature)
        else:
            concrete = torch.sigmoid(log_gate)

        stretched_concrete = concrete * (high - low) + low
        clipped_concrete = torch.clamp(stretched_concrete, 0, 1)
        concrete_list = clipped_concrete.squeeze().tolist()
        return concrete_list

    def compute_loss(self, model, inputs,return_outputs=False):
        labels = inputs['labels']

        outputs = model(**inputs)
        
        ### Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        ### We don't use .loss here since the model may return tuples instead of ModelOutput.
        cn_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        loss = cn_loss

        g1_l0_norm = 0.0
        g2_l0_norm = 0.0
        if self.gated_lambda != 0:
            for name, param in model.named_parameters():
                if "log_g1" in name and param.requires_grad:
                    g1_l0_norm += self.get_penalty(param)
                    g1_gates = self.get_gates(log_gate=param, is_train=True)
                    self.g1_prop.append(g1_gates)
                if "log_g2" in name and param.requires_grad:
                    g2_l0_norm += self.get_penalty(param)
                    g2_gates = self.get_gates(log_gate=param, is_train=True)
                    self.g2_prop.append(g2_gates)

            loss = loss + (self.gated_lambda) * g1_l0_norm / (self.model.config.num_hidden_layers * self.model.config.num_attention_heads) + (1 - self.gated_lambda) * g2_l0_norm / (self.model.config.num_hidden_layers * self.model.config.num_attention_heads)
        
        # Update gate_lambda using the scheduler
        if self.gate_scheduler == "linear":
            self.gated_lambda = self.lambda_scheduler.get_lambda()
        elif self.gate_scheduler == "cyclic":
            self.gated_lambda = self.lambda_scheduler.get_lambda()
        elif self.gate_scheduler == "perform":
            self.gated_lambda = self.lambda_scheduler.get_lambda(performance_improvement=loss - self.last_loss)
            self.last_loss = loss
        elif self.gate_scheduler == "expon":
            self.gated_lambda = self.lambda_scheduler.get_lambda()

        if return_outputs:
            return loss,outputs
        else:
            return loss
    
    # def test(self, fname, task, subtask, eval_dataset=None, model_name=None):
    #     self.model.eval()
    #     self.args.prediction_loss_only = False
    #     self.tokenizer.add_eos_token = False
    #     if not os.path.exists(fname):
    #         os.makedirs(fname)
    #     if task == "commonsense":
    #         evaluate_common_reason(eval_dataset=eval_dataset, task=task, subtask=subtask, model_name=model_name, model=self.model, tokenizer=self.tokenizer, fname=fname)
    #     elif task == "mmlu_pro":
    #         evaluate_mmlu_pro(eval_dataset=eval_dataset, task=task, subtask=subtask, model_name=model_name, model=self.model, tokenizer=self.tokenizer, fname=fname)
    #     elif task == "gem":
    #         evaluate_gem(eval_dataset=eval_dataset, task=task, subtask=subtask, model_name=model_name, model=self.model, tokenizer=self.tokenizer, fname=fname)
    #     else:
    #         print("new")
        
        ## probability record during training
        # os.makedirs(f"{fname}/{task}/{model_name.split('/')[-1]}", exist_ok=True)
        # with open(os.path.join(fname, task, model_name.split('/')[-1], f"{subtask}_g1.txt"), "w", encoding="utf-8") as file_g1:
        #     for row in self.g1_prop:
        #         file_g1.write('\t'.join(map(str, row)) + '\n')
        # with open(os.path.join(fname, task, model_name.split('/')[-1], f"{subtask}_g2.txt"), "w", encoding="utf-8") as file_g2:
        #     for row in self.g2_prop:
        #         file_g2.write('\t'.join(map(str, row)) + '\n')