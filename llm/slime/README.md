# Slime: Distributed RL Training for LLMs

[Slime](https://github.com/THUDM/slime) is an advanced post-training framework for LLMs that supports distributed reinforcement learning with Megatron-LM and SGLang, enabling efficient RL scaling across multiple algorithms (GRPO, GSPO, PPO, Reinforce++).

## Why SkyPilot + Slime?

SkyPilot makes distributed RL training **easy and cost-effective**:
- **Get GPUs instantly** across clouds and Kubernetes
- **3x cheaper** with managed spot instances  
- **Zero setup** - handles distributed infrastructure automatically
- **Docker-based** - pre-configured environment with all dependencies

## Prerequisites

### 1. Model Weight Conversion (Required)

Slime requires converting HuggingFace model weights to Megatron's `torch_dist` format before training. This is a **one-time setup step** that should be done locally or on a small instance to avoid expensive GPU time.

```bash
# Inside slimerl/slime:latest Docker container
cd /root/slime

# Load model configuration
source scripts/models/qwen3-4B.sh

# Convert weights (replace paths as needed)
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-4B \
    --save /root/Qwen3-4B_torch_dist
```

**Important**: Upload the converted `torch_dist` weights to cloud storage (S3, GCS, etc.) before launching training. The YAML expects these weights to be accessible.

### 2. Dataset Preparation

Download the training dataset:
```bash
pip install -U huggingface_hub

# Training dataset
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

# Evaluation dataset
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024
```

Upload to cloud storage and update the YAML with the paths.

## Quick Start

Before launching, you need to set the following environment variables or update them in the YAML:

```bash
export CHECKPOINT_BUCKET_NAME=your-bucket-name
export TORCH_DIST_WEIGHTS_PATH=s3://your-bucket/qwen3-4b-torch-dist  # Path to converted weights
export DATASET_PATH=s3://your-bucket/dapo-math-17k  # Training dataset
export EVAL_DATASET_PATH=s3://your-bucket/aime-2024  # Evaluation dataset
```

Launch a single-node RLHF training job on 4 GPUs:
```bash
sky launch -c slime llm/slime/slime.yaml
```

Monitor training progress:
```bash
sky logs slime
```

Access Ray dashboard:
```bash
sky status --endpoint 8265 slime
```

<p align="center">
  <img src="https://github.com/THUDM/slime/raw/main/docs/assets/slime-arch.png" alt="Slime Architecture" width="90%"/>
</p>
<p align="center"><i>Slime's three-module architecture: Training (Megatron), Rollout (SGLang), and Buffer (Ray)</i></p>

## Key Features

The example trains Qwen3-4B on the dapo-math-17k dataset using GRPO:
- **Single-node training** with 4 GPUs (colocated mode)
- **Docker-based setup** with all dependencies pre-configured
- **Checkpoint persistence** to cloud storage for resumption
- **Customizable models and datasets** via environment variables

## Optional: Enable W&B for Training Visualization

To track training curves and metrics in Weights & Biases:
```bash
# 1. Set your W&B API key locally
export WANDB_API_KEY=your-api-key

# 2. Launch with the secret flag
sky launch -c slime llm/slime/slime.yaml --secret WANDB_API_KEY

# 3. W&B logging is enabled by default in the YAML
```

## Advanced Usage

### 💰 Use Spot Instances for 3x Cost Savings

```bash
sky jobs launch -n slime-job llm/slime/slime.yaml
```
Training automatically resumes from checkpoints if preempted.

### 🔧 Customize Training Configuration

Train with different model sizes:
```bash
# For larger models, adjust resources
sky launch -c slime llm/slime/slime.yaml \
  --gpus A100-80GB:8
```

Modify training parameters:
```bash
sky launch -c slime llm/slime/slime.yaml \
  --env LEARNING_RATE=5e-6 \
  --env GLOBAL_BATCH_SIZE=512
```

### 📊 Monitor Training

View Ray dashboard for distributed job monitoring:
```bash
# Get dashboard URL
sky status --endpoint 8265 slime

# SSH tunnel for local access
ssh -L 8265:localhost:8265 slime
# Then visit http://localhost:8265
```

## Understanding the Setup

1. **Docker container**: Runs `slimerl/slime:latest` with all dependencies pre-installed
2. **Ray cluster**: Single-node Ray head for job orchestration
3. **Training job**: Submitted to Ray with Megatron (training) + SGLang (inference/rollout)
4. **Colocated mode**: Training and inference run on the same GPUs for efficiency

## Troubleshooting

- **OOM errors**: Reduce `global-batch-size` or `max-tokens-per-gpu` in the YAML
- **Missing torch_dist weights**: Ensure you completed the weight conversion prerequisite
- **Dataset not found**: Verify dataset paths in cloud storage are correctly mounted
- **Ray connection issues**: Ensure port 8265 is not blocked
- **Docker permission errors**: The container runs with appropriate GPU access via `--gpus all`

## Learn More

- [Slime Documentation](https://github.com/THUDM/slime/blob/main/docs/en/get_started/quick_start.md)
- [Slime GitHub Repository](https://github.com/THUDM/slime)
- [SkyPilot Docker Guide](https://docs.skypilot.co/en/latest/examples/docker-containers.html)
- [Weight Conversion Guide](https://github.com/THUDM/slime/blob/main/docs/en/get_started/quick_start.md#model-weight-conversion)
