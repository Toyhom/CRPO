#!/bin/bash
conda activate easyr1

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export TQDM_DISABLE=1

cd EasyR1

base_model_name="Qwen3-8B"
rl_type="crpo"
work_name="ours"
model_name=${base_model_name}_${rl_type}_${work_name}
Param="default"

MODEL_PATH=./models/${base_model_name}  # replace it with your local file path

export TENSORBOARD_DIR=./tensorboard_logs/${model_name}_${Param}_multi

echo $MODEL_PATH

val_freq=10
save_freq=-1


set -o pipefail  
python3 -m verl.trainer.main \
    config=./script/config_crpo.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.save_checkpoint_path=./result/${model_name}_${Param} \
    trainer.project_name=${model_name} \
    trainer.experiment_name=${model_name}_${Param} \
    trainer.n_gpus_per_node=8 \
    trainer.total_epochs=6 \
    trainer.val_freq=$val_freq \
    trainer.save_freq=$save_freq \
    trainer.default_local_dir=$TENSORBOARD_DIR \
    algorithm.adv_estimator=$rl_type \
    algorithm.reward_scaler=False \
    algorithm.kl_type=entropy_aware_role \
    algorithm.anchor_n=1 \
    worker.rollout.anchor_n=1 \
    worker.rollout.tensor_parallel_size=4 \
    worker.rollout.gpu_memory_utilization=0.5 \
    worker.actor.global_batch_size=32 \
    worker.actor.fsdp.fsdp_size=8 \
    worker.ref.fsdp.fsdp_size=8 \
    worker.actor.optim.lr_warmup_ratio=0.05 \
    data.rollout_batch_size=256 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.model.enable_gradient_checkpointing=False \
    2>&1 | tee "./log/${model_name}_${Param}.out"
set +o pipefail

target_dir=./result/${model_name}_${Param}
last_folder=$(find "$target_dir" -maxdepth 1 -type d ! -name "." ! -name "$(basename "$target_dir")" | sort | tail -n 1)
echo $last_folder

python3 scripts/model_merger.py --local_dir ${last_folder}/actor --hf_source_dir ./models/${base_model_name}

