#!/bin/bash
export PYTHONPATH="/home/mai-llm-train-service/LLaMA-Factory/src:$PYTHONPATH"
merge_options="  \
        --model_name_or_path  /home/mai-llm-train-service/qwen/qwen2_0.5B_instruct\
        --adapter_name_or_path  /home/mai-llm-train-service/LLaMA-Factory/saves/qwen/lora/sft_0.5B_0730\
        --template  qwen\
        --finetuning_type  lora\
        --export_dir  /home/mai-llm-train-service/qwen/Wedoctor-Q-0.5B-20240731\
        --export_device cpu\
        --export_legacy_format false\
        "


run_cmd="python3 src/llamafactory/export.py ${merge_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
