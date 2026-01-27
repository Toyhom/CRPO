# CRPO

This repository contains the training script and configuration for fine-tuning the **Qwen3-8B** model using **CRPO** for role-playing tasks. The implementation leverages the `verl` library (EasyR1).

## Overview

The training pipeline uses Ray for distributed training, supporting Tensor Parallelism and Fully Sharded Data Parallel (FSDP). It is designed to optimize role-playing performance by balancing task rewards and style consistency.

*   **Base Model:** Qwen3-8B
*   **Algorithm:** CRPO (`adv_estimator="crpo"`)
*   **KL Control:** Entropy-Aware Role (`kl_type="entropy_aware_role"`)
*   **Framework:** EasyR1 / verl

## Requirements

*   EasyR1 (verl)
*   NVIDIA GPUs (Script configured for 8 GPUs/node)

## Directory Structure

```
.
├── EasyR1/                 # Core library (verl)
├── script/
│   └── config.yaml         # Training configuration
├── train/
│   └── run_rl_qwen3_8b_crpo.sh  # Main training script
├── models/                 # Pre-trained models
├── result/                 # Checkpoints and results
└── log/                    # Execution logs
```

## Usage

1.  **Install EasyR1 (verl):**
    ```bash
    cd EasyR1
    pip install -e .
    ```

2.  **Configure Paths:**
    Edit `train/run_rl_qwen3_8b_crpo.sh` to match your local paths. Ensure the following variables point to valid locations:
    *   `MODEL_PATH`: Path to the base Qwen3-8B model.
    *   `config`: Path to the YAML configuration file.
    *   Export paths for logs and results (e.g., `TENSORBOARD_DIR`, `SWANLAB_LOG_DIR`).

3.  **Run Training:**
    ```bash
    bash train/run_rl_qwen3_8b_crpo.sh
    ```

## Outputs

*   **Checkpoints:** Saved in the directory specified by `trainer.save_checkpoint_path`.
*   **Logs:** Standard output logs are saved to `log/${model_name}_${Param}.out`.
*   **Tensorboard:** Logs are saved to `tensorboard_logs/`.
