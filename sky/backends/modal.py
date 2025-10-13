"""Modal execution script for SkyPilot backend.

This script is invoked by the Modal backend to execute tasks on Modal's
serverless infrastructure. It reads task configuration from a YAML file
and creates the appropriate Modal app, image, and function.

Uses Pythonic Modal patterns:
- Chained image setup (.from_registry() → .apt_install() → .pip_install() → .env() → .add_local_dir())
- Runtime credential setup (GCP, etc.)
- GPU-dependent package installation at runtime
- Torchrun support for distributed training

Usage:
    python modal.py <task_config.yaml>
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load task configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_backend_config() -> Dict[str, Any]:
    """Load Modal backend configuration from multiple locations."""
    _HERE = Path(__file__).resolve().parent
    _CANDIDATE_PATHS = [
        _HERE / 'modal.yaml',
        Path.home() / '.sky' / 'modal.yaml',
        Path('modal.yaml'),
    ]
    _ENV_PATH = os.environ.get('MODAL_CONFIG_PATH')
    if _ENV_PATH:
        _CANDIDATE_PATHS.insert(0, Path(_ENV_PATH))
    
    for _p in _CANDIDATE_PATHS:
        if _p.exists():
            with _p.open('r') as f:
                config = yaml.safe_load(f) or {}
            return config
    
    return {'gpu': 'A100:1', 'secrets': []}


def create_modal_image(
    backend_config: Dict[str, Any],
    task_config: Dict[str, Any]
) -> modal.Image:
    """Create Modal image with Pythonic chained setup."""
    image_config = backend_config.get('image', {})
    base_image = image_config.get('base', 'pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel')
    
    image = modal.Image.from_registry(base_image)
    
    apt_packages = image_config.get('apt_packages', [])
    if apt_packages:
        image = image.apt_install(*apt_packages)
    
    pip_packages = image_config.get('pip_packages', [])
    if pip_packages:
        image = image.pip_install(*pip_packages)
    
    env_vars = task_config.get('env', {})
    env_for_image = {k: str(v) for k, v in env_vars.items() if v is not None}
    if env_for_image:
        image = image.env(env_for_image)
    
    workdir = task_config.get('workdir')
    if workdir and Path(workdir).exists() and Path(workdir).is_dir():
        image = image.add_local_dir(str(workdir), '/workspace/code')
    
    setup_commands = task_config.get('setup', [])
    for cmd in setup_commands:
        if 'cuda' not in cmd.lower() and 'gpu' not in cmd.lower() and 'torch' not in cmd.lower():
            image = image.run_commands(cmd)
    
    return image


def _try_secret(name: str) -> Optional[modal.Secret]:
    """Try to load a Modal secret, return None if not found."""
    try:
        return modal.Secret.from_name(name)
    except Exception:
        return None


def create_modal_secrets(secret_names: List[str]) -> List[modal.Secret]:
    """Create list of Modal secrets using safe loading."""
    return list(filter(None, [_try_secret(n) for n in secret_names]))


def create_file_mounts(file_mounts: Dict[str, Any]) -> Dict[str, modal.Volume]:
    """Create Modal volumes for file mounts."""
    volumes = {}
    for mount_path, mount_config in file_mounts.items():
        if isinstance(mount_config, dict):
            volume_name = mount_config.get('name', f'skypilot-{mount_path.replace("/", "-")}')
            
            try:
                volume = modal.Volume.from_name(volume_name, create_if_missing=True)
                volumes[mount_path] = volume
                print(f'Mounted volume {volume_name} at {mount_path}')
            except Exception as e:
                print(f'Warning: Could not create volume for {mount_path}: {e}')
    
    return volumes


def setup_gcp_credentials():
    """Set up GCP credentials from Modal secret at runtime."""
    try:
        gcp_creds = os.environ.get('GCP_CREDENTIALS')
        if gcp_creds:
            creds_path = '/tmp/gcp_credentials.json'
            with open(creds_path, 'w') as f:
                f.write(gcp_creds)
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_path
            print(f'✅ GCP credentials set up at {creds_path}')
        else:
            print('⚠️ No GCP credentials found in environment')
    except Exception as e:
        print(f'Warning: Could not set up GCP credentials: {e}')


def install_runtime_dependencies(setup_commands: List[str]):
    """Install GPU-dependent packages at runtime."""
    gpu_dependent_commands = [
        cmd for cmd in setup_commands
        if any(keyword in cmd.lower() for keyword in ['cuda', 'gpu', 'torch', 'grouped_gemm'])
    ]
    
    for cmd in gpu_dependent_commands:
        print(f'Installing GPU-dependent package: {cmd}')
        try:
            subprocess.run(['/bin/bash', '-c', cmd], check=True)
            print(f'✅ Installed: {cmd}')
        except Exception as e:
            print(f'Warning: Could not install {cmd}: {e}')


def main():
    parser = argparse.ArgumentParser(description='Execute SkyPilot task on Modal')
    parser.add_argument('config', help='Path to task configuration YAML file')
    args = parser.parse_args()
    
    task_config = load_config(args.config)
    backend_config = load_backend_config()
    
    gpu_spec = task_config.get('gpu', backend_config.get('gpu', 'A100:1'))
    secrets = task_config.get('secrets', backend_config.get('secrets', []))
    run_commands = task_config.get('run', [])
    setup_commands = task_config.get('setup', [])
    workdir = task_config.get('workdir', '/workspace/code')
    file_mounts = task_config.get('file_mounts', {})
    timeout = task_config.get('timeout', backend_config.get('timeout', 86400))
    use_torchrun = task_config.get('use_torchrun', False)
    
    app = modal.App('skypilot-modal-task')
    
    image = create_modal_image(backend_config, task_config)
    modal_secrets = create_modal_secrets(secrets)
    volumes = create_file_mounts(file_mounts)
    
    @app.function(
        gpu=gpu_spec,
        image=image,
        secrets=modal_secrets if modal_secrets else None,
        volumes=volumes if volumes else None,
        timeout=timeout,
    )
    def run_task():
        """Execute the SkyPilot task commands."""
        os.chdir(workdir)
        
        setup_gcp_credentials()
        install_runtime_dependencies(setup_commands)
        
        env_vars = task_config.get('env', {})
        for key, value in env_vars.items():
            if value is None:
                continue
            os.environ[key] = str(value)
        
        if use_torchrun and run_commands:
            n_gpus = gpu_spec.split(':', 1)[1] if ':' in gpu_spec else '1'
            
            script_path = run_commands[0]
            additional_args = run_commands[1:] if len(run_commands) > 1 else []
            
            cmd = [
                'torchrun',
                f'--nproc-per-node={n_gpus}',
                script_path,
                *additional_args,
            ]
            
            print(f'\n=== Executing (torchrun): {" ".join(cmd)} ===')
            result = subprocess.run(cmd, check=False)
            return result.returncode
        else:
            for cmd in run_commands:
                print(f'\n=== Executing: {cmd} ===')
                result = subprocess.run(
                    ['/bin/bash', '-c', cmd],
                    check=False,
                )
                if result.returncode != 0:
                    print(f'Command failed with return code {result.returncode}')
                    return result.returncode
        
        return 0
    
    with app.run():
        return_code = run_task.remote()
        sys.exit(return_code if return_code is not None else 0)


if __name__ == '__main__':
    main()
