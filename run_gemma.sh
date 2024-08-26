#!/bin/bash
# 本地路由
CURRENT_DIR="$( cd "$( dirname "$0" )" && pwd )"
export PYTHONPATH=$PYTHONPATH:${CURRENT_DIR}/src
NNODES=1 # Node数
NODE_RANK=0
GPUS_PER_NODE=4 # Node卡数
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK"

run_cmd="torchrun $DISTRIBUTED_ARGS src/llamafactory/launcher.py examples/train_full/gemma_full_sft_ds3.yaml"

echo ${run_cmd}
eval ${run_cmd}
set +x