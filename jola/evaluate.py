import os
import torch
import json
import re
import copy
from tqdm import tqdm
import gem_metrics

from transformers import GenerationConfig

def extract_common_reason_answer(sentence: str, task) -> float:
    if task == 'BoolQ':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif task == 'PIQA':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif task in ['SIQA', 'ARC-c', 'ARC-e', 'OBQA']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif task == 'HellaSwag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif task == 'WinoGrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]

def extract_mmlu_pro_answer(sentence: str) -> float:
    sentence_ = sentence.strip()
    pred_answers = re.findall(r'A|B|C|D|E|F|G|H|I|J', sentence_)
    if not pred_answers:
        return ""
    return pred_answers[0]

def extract_gem_answer(sentence: str) -> str:
    sentence_ = sentence.strip()
    pred_answers = sentence_.replace(".", "")
    return pred_answers


def create_batch(dataset, batch_size):
    batches = []
    num_batch = len(dataset)//batch_size if len(dataset) % batch_size == 0 else len(dataset)//batch_size + 1
    for i in range(num_batch):
        batch = dataset[i*batch_size: min((i+1)*batch_size, len(dataset))]
        batches.append(batch)
    return batches

def evaluate_single(model, tokenizer, instructions, input=None, temperature=0.1, top_p=0.75, top_k=40, num_beams=4, max_new_tokens=16, **kwargs,):
    prompts = instructions
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(model.device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences
    outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
    outputs = [o.split("### Response:")[1].strip().split("\n")[0] for o in outputs]
    return outputs


def evaluate_common_reason(eval_dataset, task, subtask, model_name, model, tokenizer, fname, batch_size=1):
    tokenizer.padding_side = 'left'
    batches = create_batch(eval_dataset, batch_size)
    total = len(batches)
    pbar = tqdm(total=total)
    output_data = []
    current = 0
    correct = 0

    for idx, batch in enumerate(batches):
        current += len(batch['prompt'])
        instructions = batch['prompt']
        labels=batch['target_text']
        outputs = evaluate_single(model, tokenizer, instructions)

        for data_label, output, instuction in zip(labels, outputs, instructions):
            label = data_label # 
            flag = False
            predict = extract_common_reason_answer(output, subtask)
            if label == predict:
                correct += 1
                flag = True
            new_data = {}
            new_data['instuction'] = instuction
            new_data['answer'] = label
            new_data['output_pred'] = output
            new_data['pred'] = predict
            new_data['flag'] = flag
            output_data.append(new_data)
            
        print('---------------')
        print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / current}')
        print('---------------')
        pbar.update(1)
    
    output_data.append({"acc": f"{correct / current}"})
    os.makedirs(f"{fname}/{task}/{model_name.split('/')[-1]}", exist_ok=True)
    with open(os.path.join(fname, task, f"{model_name.split('/')[-1]}", f"{subtask}.json"), 'w+') as f:
        json.dump(output_data, f, indent=4)
    
    pbar.close()
    print('\n')
    print('test finished')


def evaluate_mmlu_pro(eval_dataset, task, subtask, model_name, model, tokenizer, fname, batch_size=1):
    tokenizer.padding_side = 'left'
    batches = create_batch(eval_dataset, batch_size)
    total = len(batches)
    pbar = tqdm(total=total)
    output_data = []
    current = 0
    correct = 0

    for idx, batch in enumerate(batches):
        current += len(batch['prompt'])
        instructions = batch['prompt']
        labels=batch['target_text']
        outputs = evaluate_single(model, tokenizer, instructions)

        for data_label, output, instuction in zip(labels, outputs, instructions):
            label = data_label # 
            flag = False
            predict = extract_mmlu_pro_answer(output)
            if label == predict:
                correct += 1
                flag = True
            new_data = {}
            new_data['instuction'] = instuction
            new_data['answer'] = label
            new_data['output_pred'] = output
            new_data['pred'] = predict
            new_data['flag'] = flag
            output_data.append(new_data)
            
        print('---------------')
        print(f'\rtest:{idx + 1}/{total} | accuracy {correct}  {correct / current}')
        print('---------------')
        pbar.update(1)
    
    output_data.append({"acc": f"{correct / current}"})
    os.makedirs(f"{fname}/{task}/{model_name.split('/')[-1]}", exist_ok=True)
    with open(os.path.join(fname, task, f"{model_name.split('/')[-1]}", f"{subtask}.json"), 'w+') as f:
        json.dump(output_data, f, indent=4)
    
    pbar.close()
    print('\n')
    print('test finished')


def evaluate_gem(eval_dataset, task, subtask, model_name, model, tokenizer, fname, batch_size=1):
    tokenizer.padding_side = 'left'
    batches = create_batch(eval_dataset, batch_size)
    total = len(batches)
    pbar = tqdm(total=total)
    output_data = []
    
    predict_list = []
    reference_list = []

    for idx, batch in enumerate(batches):
        instructions = batch['prompt']
        labels=batch['target_text']
        outputs = evaluate_single(model, tokenizer, instructions, max_new_tokens=64)

        for data_label, output, instuction in zip(labels, outputs, instructions):
            reference_list.append(data_label)
            pred_clean = extract_gem_answer(output)
            predict_list.append(pred_clean)

            new_data = {}
            new_data["instuction"] = instuction
            new_data["answer"] = data_label
            new_data["pred"] = pred_clean
            output_data.append(new_data)

        pbar.update(1)
    
    ## metrics computing
    preds = gem_metrics.texts.Predictions(predict_list)
    refs = gem_metrics.texts.References(reference_list)  # input may be list of lists for multi-ref

    res_metrics = gem_metrics.compute(preds, refs, metrics_list=['bleu', 'rouge', 'nist', 'chrf', 'cider'])  # add list of desired metrics here
    ###
    print(f"#############------Metric in {task}-------###############")
    print(f"NIST: {res_metrics['nist']}")
    print(f"chrf: {res_metrics['chrf']}")
    print(f"chrf+: {res_metrics['chrf+']}")
    print(f"chrf++: {res_metrics['chrf++']}")
    print(f"Rouge1: {res_metrics['rouge1']['fmeasure']}")
    print(f"Rouge2: {res_metrics['rouge2']['fmeasure']}")
    print(f"RougeL: {res_metrics['rougeL']['fmeasure']}")
    print(f"RougeLsum: {res_metrics['rougeLsum']['fmeasure']}")
    print(f"CIDEr: {res_metrics['CIDEr']}")
    print(f"BLEU: {res_metrics['bleu']}")
    print(f"#############------Metric in {task}-------###############")
    metrics_dict = {
        "NIST": f"{res_metrics['nist']}",
        "chrf": f"{res_metrics['chrf']}",
        "chrf+": f"{res_metrics['chrf+']}",
        "chrf++": f"{res_metrics['chrf++']}",
        "Rouge1": f"{res_metrics['rouge1']['fmeasure']}",
        "Rouge2": f"{res_metrics['rouge2']['fmeasure']}",
        "RougeL": f"{res_metrics['rougeL']['fmeasure']}",
        "RougeLsum": f"{res_metrics['rougeLsum']['fmeasure']}",
        "CIDEr": f"{res_metrics['CIDEr']}",
        "BLEU": f"{res_metrics['bleu']}"
    }

    output_data.append(metrics_dict)
    os.makedirs(f"{fname}/{task}/{model_name.split('/')[-1]}", exist_ok=True)
    with open(os.path.join(fname, task, f"{model_name.split('/')[-1]}", f"{subtask}.json"), 'w+') as f:
        json.dump(output_data, f, indent=4)
    
    pbar.close()
    print('\n')
    print('test finished')