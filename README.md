# JoLA: Joint Localization and Activation Editing for Low-Resource Fine-Tuning

## Abstract
> Parameter-efficient fine-tuning (PEFT) methods, such as LoRA, are commonly used to adapt LLMs. However, their effectiveness is often limited in low-resource scenarios with only a few hundred examples. Recent advances in interpretability research have inspired the emergence of activation editing techniques, which modify the activations of specific model components. These methods, due to their extremely small parameter counts, show promise for small datasets. However, their performance is highly dependent on identifying the correct modules to edit and often lacks stability across different datasets. In this paper, we propose Joint Localization and Activation Editing (JoLA), a method that jointly learns (1) which heads in the Transformer to edit; (2) whether the intervention should be additive, multiplicative, or both and (3) the intervention parameters themselves - vectors applied as additive offsets or multiplicative scalings to the head output. Through evaluations on three benchmarks spanning commonsense reasoning, natural language understanding, and natural language generation, we demonstrate that JoLA consistently outperforms existing methods.

## Contents
1. [Environment](#Environment)
2. [Dataset](#Dataset)
3. [Training](#Training)

## Environment
+ Transformers (>5.14.3)
+ torch (>2.5.1)
+ trl (>0.8.1)
+ datasets (>3.2.0)
+ gem_metrics (https://github.com/GEM-benchmark/GEM-metrics)

## Dataset
We evaluate on commonsense reasoning, natural language understanding and natural language generation benchmarks.
+ Commonsense Reasoning
    - The same as [Hu et al., 2023](https://aclanthology.org/2023.emnlp-main.319/)
    - 8 Tasks: ARC-c / ARC-e / BoolQ / HellaSwag / OBQA / PIQA / SIQA / WinoGrande
    - [Download Link](https://github.com/AGI-Edgerunners/LLM-Adapters)
+ Natural Language Understanding
    - We use MMLU-Pro Benchmark ([Wang et al., 2024](https://arxiv.org/abs/2406.01574))
    - 14 Domains: Biology, Business, Chemistry, Computer Science, Economics, Engineering, Health, History, Law, Math, Philosophy, Physics, Psychology, and Others
    - [Download Link](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
+ Natural Language Generation
    - We use [GEM Benchmark](https://gem-benchmark.com/) ([Gehrmann et al., 2022](https://arxiv.org/abs/2206.11249))
    - We use 4 tasks: CommonGen / E2E_NLG / Web_NLG / Xsum
    - [Download Link](https://huggingface.co/datasets/GEM/gem)
    - Prompt Template from [PromptSource](https://github.com/bigscience-workshop/promptsource)

## Training
+ Commonsense Reasoning
```
TASK=commonsense
MODEL_NAME=llama3.1_8B_chat
SEED=42
TRAIN_SIZE=200
GATE_SCHEDULE=expon

# set the Task List
TASK_LIST=("ARC-c" "ARC-e" "BoolQ" "HellaSwag" "OBQA" "PIQA" "SIQA" "WinoGrande")

for subtask in "${TASK_LIST[@]}"
do
    echo "Process ${TASK} ${subtask}"
    python $PRJ_PATH/run_trainer.py \
    --task $TASK \
    --subtask $subtask \
    --hf_cache_dir $CACHE_PATH \
    --data_path ${DATA_PATH}/ \
    --base_model_name $MODEL_NAME \
    --lr 5e-3 \
    --train_batch 8 \
    --num_epoch 20 \
    --output_dir ${SAVE_PATH}/model/${MODEL_NAME}/${TASK}_${subtask} \
    --run_mode train \
    --output_file_name ${SAVE_PATH}/res \
    --applied_module attention \
    --save_strategy best \
    --gate_scheduler $GATE_SCHEDULE \
    --eval_batch 1 \
    --train_size $TRAIN_SIZE \
    --seed $SEED

done
```
+ same setting in MMLU_Pro and GEM tasks
