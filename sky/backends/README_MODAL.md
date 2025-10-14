# Modal Backend for SkyPilot

The Modal backend allows you to run SkyPilot tasks on [Modal's](https://modal.com) serverless infrastructure instead of traditional cloud VMs. This is particularly useful for GPU-intensive workloads that benefit from Modal's fast cold starts and automatic scaling.

## Prerequisites

1. **Modal Account**: Sign up at https://modal.com
2. **Modal CLI**: Install the Modal Python package
   ```bash
   pip install modal
   ```
3. **Authentication**: Set up Modal credentials
   ```bash
   modal setup
   ```

## Configuration

### 1. Backend Configuration (modal.yaml)

The Modal backend reads configuration from `sky/backends/modal.yaml`. This file is already included in the repository with default settings:

```yaml
gpu: "A100-80GB:8"
secrets:
  - wandb-secret
  - gcp-credentials
image:
  base: "python:3.11-slim"
  apt_packages:
    - git
    - curl
  pip_packages:
    - torch
    - transformers
timeout: 86400
```

You can override these settings by creating `~/.sky/modal.yaml` or setting the `MODAL_CONFIG_PATH` environment variable.

### 2. Configure Modal Secrets

Modal secrets are used to inject sensitive credentials (API keys, authentication tokens) into your tasks. Create secrets in your Modal account:

```bash
# Example: Create a WandB secret
modal secret create wandb-secret WANDB_KEY=your_wandb_api_key_here

# Example: Create GCP credentials secret
modal secret create gcp-credentials GCP_CREDENTIALS="$(cat path/to/credentials.json)"

# Example: Create Hugging Face token secret
modal secret create hf-token HF_TOKEN=your_hf_token_here
```

List your configured secrets:
```bash
modal secret list
```

### 3. Update Backend Configuration

Edit `~/.sky/modal.yaml` to reference your secrets:

```yaml
gpu: "A100-80GB:8"  # Adjust GPU type and count as needed
secrets:
  - wandb-secret
  - gcp-credentials
  - hf-token
```

## Usage

### Basic Usage

Launch a task using the Modal backend:

```bash
sky launch --backend modal your_task.yaml
```

Or explicitly specify the backend in your task YAML:

```yaml
# your_task.yaml
name: my-modal-task

resources:
  accelerators: {A100-80GB:8}

envs:
  WANDB_KEY: null  # Will be injected from Modal secret

setup: |
  pip install torch transformers datasets

run: |
  python train.py
```

### File Mounts and Persistent Storage

The Modal backend translates SkyPilot's `file_mounts` to Modal Volumes. Volumes are automatically created if they don't exist:

```yaml
file_mounts:
  /workspace/checkpoints:
    name: my-checkpoints
    store: gcs  # Note: store type is ignored; Modal uses its own storage
    persistent: true
  /workspace/data:
    name: my-data
    persistent: true
```

Each mount creates or references a Modal Volume with the specified name. Data persists across runs when `persistent: true`.

### Environment Variables

Environment variables can be specified in your task YAML:

```yaml
envs:
  HF_HUB_ENABLE_HF_TRANSFER: "1"
  WANDB_KEY: null  # null means read from Modal secret
  MY_CONFIG: "production"
```

Variables with `null` values are expected to be provided by Modal secrets. Other variables are passed directly to the Modal function.

## Architecture

The Modal backend consists of three main components:

1. **modal_backend.py**: Main backend implementation that handles task orchestration
2. **modal.yaml**: Configuration file for GPU specs, secrets, and image settings
3. **modal.py**: Execution script that creates Modal apps and runs tasks

### Execution Flow

1. User runs `sky launch --backend modal task.yaml`
2. `ModalBackend` reads `modal.yaml` for backend configuration
3. Backend translates the SkyPilot task to a Modal-compatible configuration
4. Backend invokes `modal.py` with the task configuration
5. `modal.py` creates a Modal app with appropriate image, GPU, and secrets
6. Setup commands are baked into the Modal image
7. Run commands are executed in the Modal function
8. File mounts are handled via Modal Volumes

## Comparison with Cloud VM Backend

| Feature | Cloud VM Backend | Modal Backend |
|---------|-----------------|---------------|
| Provisioning | ~2-5 minutes | ~30 seconds |
| GPU Options | All cloud providers | Modal's GPU offerings |
| Persistent Storage | Cloud storage (S3, GCS, etc.) | Modal Volumes |
| Networking | Full VM networking | Serverless function |
| Multi-node | Supported | Single-node with multiple GPUs |
| Cost Model | Per-minute VM pricing | Per-second function pricing |

## Limitations

1. **Single-node only**: Multi-node distributed training is not currently supported. Use multiple GPUs on a single node instead.
2. **Storage**: File mounts use Modal Volumes instead of direct cloud storage mounting (GCS, S3). Data must be copied to/from volumes.
3. **Networking**: Limited compared to full VMs. No direct SSH access or custom port forwarding.
4. **Setup commands**: Run during image build time, not at runtime. Changes to setup require rebuilding the image.

## Troubleshooting

### "Secret not found" error

Make sure you've created the secret in your Modal account:
```bash
modal secret list
modal secret create secret-name KEY=value
```

### GPU type not available

Check Modal's available GPU types and update your `modal.yaml`:
```bash
modal gpu list
```

### Image build failures

Check that your setup commands are compatible with the base image. You may need to adjust the base image in `modal.yaml`.

### Volume mount issues

Ensure volume names don't contain invalid characters. Use lowercase letters, numbers, and hyphens only.

## Example: Running Slime Training

See `examples/modal_slime_task.yaml` for a complete example of running the Slime RL training workflow on Modal.

## Launching Slime Training on Modal

This section provides step-by-step instructions for running Slime RL training on Modal's serverless infrastructure.

### Prerequisites

1. **Modal account and CLI setup:**
   ```bash
   pip install modal
   modal setup
   ```

2. **SkyPilot with Modal backend:**
   ```bash
   pip install "skypilot[aws]"  # or your preferred cloud
   ```

3. **Configure Modal secrets:**
   ```bash
   # WandB API key for experiment tracking
   modal secret create wandb-secret WANDB_API_KEY=your_wandb_key_here
   
   # Hugging Face token for model downloads
   modal secret create hf-token HF_TOKEN=your_hf_token_here
   
   # (Optional) GCP credentials for cloud storage
   modal secret create gcp-credentials GCP_CREDENTIALS="$(cat path/to/credentials.json)"
   ```

4. **Update Modal backend configuration** (`~/.sky/modal.yaml`):
   ```yaml
   gpu: "H100:8"
   secrets:
     - wandb-secret
     - hf-token
     - gcp-credentials  # if using GCS
   
   image:
     base: "pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel"
     apt_packages:
       - git
       - tmux
       - htop
       - curl
     pip_packages:
       - transformers
       - datasets
       - huggingface-hub
       - wandb
       - google-cloud-storage
   
   timeout: 86400
   ```

### Quick Start: GLM-4 9B Training

**Step 1: Launch training job**
```bash
sky launch --backend modal -c slime-glm4-9b examples/modal_slime_glm4_9b.yaml
```

This command will:
- Create a Modal app with H100:8 GPUs
- Build a Docker image with all dependencies
- Download the GLM-4 9B model from Hugging Face
- Download the training datasets
- Start the RL training job
- Save checkpoints to Modal Volume

**Step 2: Monitor training progress**

Since Modal functions are ephemeral and don't provide SSH access, you can monitor progress by:

1. **Check Modal dashboard:** Visit https://modal.com/apps to see your running function
2. **View logs in real-time:** Modal CLI automatically streams logs during execution
3. **Check WandB:** If configured, training metrics will be logged to Weights & Biases

**Step 3: Access checkpoints**

Checkpoints are saved to a Modal Volume named `slime-glm4-checkpoints-modal`. To access them:

```bash
# List volumes
modal volume ls

# Download checkpoint data
modal volume get slime-glm4-checkpoints-modal /local/path/to/save
```

### Customizing the Configuration

You can customize the training by modifying environment variables or the YAML file:

**Option 1: Environment variables (for quick experiments)**
```bash
export HF_MODEL_ID=your-org/your-model-id
export MODEL_DIR=/root/your-model-name
export WANDB_PROJECT=my-experiment

sky launch --backend modal -c my-experiment examples/modal_slime_glm4_9b.yaml
```

**Option 2: Custom YAML (for reproducible experiments)**
```bash
# Copy and modify the example
cp examples/modal_slime_glm4_9b.yaml my_custom_training.yaml

# Edit my_custom_training.yaml with your preferences:
# - Change HF_MODEL_ID to your model
# - Adjust training hyperparameters in the run section
# - Modify GPU count if needed

sky launch --backend modal -c my-experiment my_custom_training.yaml
```

### Model Selection Examples

**GLM-4 9B (supported):**
```yaml
envs:
  HF_MODEL_ID: zai-org/GLM-Z1-9B-0414
  MODEL_DIR: /root/GLM-Z1-9B-0414

resources:
  accelerators: H100:8  # Single-node, 8 GPUs
```

**Other models:**
- Qwen3-4B: `HF_MODEL_ID: font-info/qwen3-4b-sft-SGLang-RL`
- Custom models: Set `HF_MODEL_ID` to your Hugging Face model ID

### Important Limitations

⚠️ **Multi-node training not supported:**
- Modal supports up to 8 GPUs per container
- The GLM-4.5 355B example (8 nodes × 64 GPUs) **cannot** run on Modal
- For multi-node training, use traditional SkyPilot backends (AWS, GCP, Azure)

⚠️ **Large model downloads:**
- Model downloads happen during image build or at runtime
- Large models (>10GB) may take significant time
- Consider pre-downloading models to a Modal Volume for faster startup

⚠️ **Storage:**
- File mounts are translated to Modal Volumes (not direct S3/GCS mounting)
- Data must be copied to/from volumes
- Volumes persist across runs when `persistent: true`

### Troubleshooting

**Issue: "Secret not found"**
```bash
# List configured secrets
modal secret list

# Create missing secret
modal secret create secret-name KEY=value
```

**Issue: "GPU not available"**
```bash
# Check available GPU types
modal gpu list

# Update ~/.sky/modal.yaml with available GPU type
```

**Issue: "Model download fails"**
- Verify HF_TOKEN is set correctly in Modal secrets
- Check model ID exists on Hugging Face: https://huggingface.co/{model-id}
- Try downloading manually first: `huggingface-cli download {model-id}`

**Issue: "Timeout during image build"**
- Large model downloads may timeout during image build
- Consider downloading models at runtime instead
- Or pre-download to Modal Volume and mount it

### Cost Optimization

1. **Use appropriate GPU types:**
   - Development/testing: T4, L4 (cheaper)
   - Production training: H100, A100 (faster)

2. **Set timeout limits:**
   ```yaml
   # In your task YAML
   timeout: 3600  # 1 hour (adjust based on training time)
   ```

3. **Monitor usage:**
   - Check Modal dashboard for compute time
   - Review WandB for training efficiency
   - Stop jobs early if not converging

### Next Steps

- Review `examples/modal_slime_glm4_9b.yaml` for full example
- Check `sky/backends/modal.yaml` for backend configuration
- Read Modal documentation: https://modal.com/docs

## Support

For Modal-specific issues, see https://modal.com/docs
For SkyPilot issues, see https://github.com/skypilot-org/skypilot
