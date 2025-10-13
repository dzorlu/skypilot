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

## Support

For Modal-specific issues, see https://modal.com/docs
For SkyPilot issues, see https://github.com/skypilot-org/skypilot
