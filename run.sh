#!/bin/bash
export PYTHONPATH="/home/mai-llm-train-service/LLaMA-Factory/src:$PYTHONPATH"
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK"

run_cmd="torchrun $DISTRIBUTED_ARGS src/llamafactory/launcher.py examples/train_lora/qwen_lora_sft_ds2.yaml"

echo ${run_cmd}
eval ${run_cmd}
set +x