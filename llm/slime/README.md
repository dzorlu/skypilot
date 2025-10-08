# Slime: RL Training for Tool-Using LLMs


[Slime](https://github.com/THUDM/slime) is an advanced post-training framework for LLMs that supports distributed reinforcement learning with Megatron-LM and SGLang. This example demonstrates training **Qwen3-4B with ReTool** - teaching the model to use Python code execution tools to solve math problems.

## Why SkyPilot + Slime?

- **Zero setup** - All dependencies in Docker container
- **Fully automated** - Downloads datasets, converts models, and starts training
- **3x cheaper** - Run on spot instances with automatic recovery
- **Cross-cloud** - Works on AWS, GCP, Azure, Lambda, and more

## Quick Start

### Prerequisites

**No dataset setup required!** The YAML automatically downloads:
- Training data: dapo-math-17k (17K math problems)
- Evaluation data: AIME-2024 (competition problems)
- Pre-trained model: qwen3-4b-sft-SGLang-RL

### Launch Training

The setup provides **two training modes**

**Option 1: Standard Training (Recommended for getting started)**

```bash
sky launch -c slime llm/slime/slime.yaml \
  --env WANDB_KEY=<your-wandb-api-key>
```

**Option 2: Async Training with tool use**

```bash
sky launch -c slime-tool llm/slime/slime-async-retool.yaml \
  --env WANDB_KEY=<your-wandb-api-key>
```

### What Gets Trained

The model learns to:
- **Use code execution tools** to solve mathematical problems
- **Generate and execute Python code** in a safe sandbox
- **Verify solutions** with reward modeling
- **Improve through RL** using the GRPO algorithm

**Configuration:**
- Model: Qwen3-4B (pre-trained SFT checkpoint)
- Dataset: 17K math problems from dapo-math-17k
- Evaluation: AIME 2024 competition problems
- GPUs: 8x A100/H100
- Algorithm: GRPO (Group Relative Policy Optimization)

### Monitor Training

Track metrics in your W&B dashboard and monitor progress:
```bash
# View training logs (replace cluster name accordingly)
sky logs slime-retool --follow   # or slime-async

# Check Ray dashboard
sky status --endpoint 8265 slime-retool
```

### Training Mode Comparison

| Feature | Standard | Async |
|---------|----------|-------|
| **Script** | `train.py` | `train_async.py` |
| **GPU usage** | 4 GPUs colocated (training + rollout share) | 4 GPUs training + 4 GPUs rollout (separate) |
| **Pattern** | Sequential: train → wait → rollout → train | Parallel: persistent worker generates while training |
| **Throughput** | Standard | **30-50% faster** |
| **Best for** | Learning, debugging, simpler setup | Production, maximum GPU utilization |

**How async works:** A persistent background worker continuously generates rollouts using separate GPUs while training happens in parallel. Training fetches already-completed samples from a queue, eliminating wait time.

## Advanced Usage

### 💰 Use Spot Instances (3x Cost Savings)

```bash
sky jobs launch -n slime-job llm/slime/slime.yaml \
  --env WANDB_KEY=<your-wandb-api-key>
```
Automatically handles preemptions and resumes from checkpoints.

### 🔧 Customize Resources

Train on different GPU types:
```bash
# Use H100 GPUs
sky launch -c slime-retool llm/slime/slime.yaml \
  --env WANDB_KEY=<your-key> --gpus H100:4

# Use A100-80GB
sky launch -c slime-retool llm/slime/slime.yaml \
  --env WANDB_KEY=<your-key> --gpus A100-80GB:4
```

### 📊 Monitor Training

Access the Ray dashboard:
```bash
# Get the dashboard URL
ENDPOINT=$(sky status --endpoint 8265 slime-retool)
echo "Ray Dashboard: http://$ENDPOINT"

# Or create SSH tunnel for local access
ssh -L 8265:localhost:8265 slime-retool
# Then visit http://localhost:8265
```

### 🎯 Using Custom Datasets (Optional)

By default, everything downloads automatically. To use your own datasets:

1. **Upload to HuggingFace or cloud storage**
2. **Modify the YAML** download commands:
   ```bash
   # Change from:
   hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/dapo-math-17k
   
   # To your dataset:
   hf download --repo-type dataset your-org/your-dataset --local-dir /root/your-dataset
   ```
3. **Update paths** in the training script section

See [ReTool README](https://github.com/THUDM/slime/blob/main/examples/retool/README.md) for more customization options.

## How It Works

The training pipeline (fully automatic):

1. **Setup Phase** (no user action required):
   - Pull Slime Docker image with all dependencies
   - Download datasets in parallel from HuggingFace:
     - `dapo-math-17k` → `/root/dapo-math-17k/`
     - `aime-2024` → `/root/aime-2024/`
     - `qwen3-4b-sft` → `/root/font-info/qwen3-4b-sft/`
   - Convert model weights to Megatron `torch_dist` format

2. **Training Phase**:
   - Start Ray cluster for distributed orchestration
   - Use custom ReTool generate function for tool-augmented responses
   - Apply GRPO for policy optimization
   
   **Standard mode (`slime.yaml` with `train.py`):**
   - Uses `--colocate` flag: training and rollout share 4 GPUs
   - Sequential: training step → wait for rollout → training step → ...
   - Simpler, easier to debug
   
   **Async mode (`slime-async-retool.yaml` with `train_async.py`):**
   - Separate GPU allocation: 4 for training, 4 for rollout
   - Persistent background worker continuously generates rollouts
   - Training fetches pre-computed samples from queue
   - Training and rollout happen in parallel for maximum throughput!

3. **Checkpointing**:
   - Standard: `/root/font-info/qwen3-4b-sft/qwen3-4b-sft-multi-turn/`
   - Async: `/root/font-info/qwen3-4b-sft/async-retool-ckpts/`
   - Automatic checkpoint resumption on restarts

## ReTool Features

The ReTool framework enables:
- **Safe code execution** in sandboxed environments
- **Tool calling** with structured XML format
- **Reward modeling** based on solution correctness
- **Multi-turn reasoning** with intermediate tool results

Example tool format:
```xml
<tools>
{"type": "function", "function": {"name": "code_interpreter", ...}}
</tools>

<tool_call>
{"name": "code_interpreter", "arguments": {"code": "print(2+2)"}}
</tool_call>
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **OOM errors** | Reduce `--max-tokens-per-gpu` in YAML or modify PERF_ARGS |
| **Ray won't start** | Script automatically cleans up old Ray processes |
| **Download fails** | Check HuggingFace access and network connectivity |
| **Slow download** | Downloads run in parallel; `HF_HUB_ENABLE_HF_TRANSFER=1` enabled |
| **Async mode needs 8 GPUs** | Yes, async uses 4 for training + 4 for rollout (separate) |
| **Standard mode GPU usage** | Standard uses 4 GPUs with `--colocate` (shared) |

## Learn More

- 📖 [Slime Documentation](https://github.com/THUDM/slime/blob/main/docs/en/get_started/quick_start.md)
- 🔧 [ReTool Example Details](https://github.com/THUDM/slime/blob/main/examples/retool/README.md)
- 🚀 [Slime GitHub Repository](https://github.com/THUDM/slime)
- 📚 [SkyPilot Docker Guide](https://docs.skypilot.co/en/latest/examples/docker-containers.html)
