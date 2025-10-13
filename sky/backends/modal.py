"""Modal execution script for SkyPilot backend.

This script is invoked by the Modal backend to execute tasks on Modal's
serverless infrastructure. It reads task configuration from a YAML file
and creates the appropriate Modal app, image, and function.

Usage:
    python modal.py <task_config.yaml>
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import modal
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load task configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_backend_config() -> Dict[str, Any]:
    """Load Modal backend configuration."""
    backend_config_path = Path(__file__).parent / 'modal.yaml'
    if backend_config_path.exists():
        with open(backend_config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def create_modal_image(
    backend_config: Dict[str, Any],
    task_config: Dict[str, Any]
) -> modal.Image:
    """Create Modal image with required dependencies."""
    image_config = backend_config.get('image', {})
    base_image = image_config.get('base', 'python:3.11-slim')
    
    image = modal.Image.from_registry(base_image)
    
    apt_packages = image_config.get('apt_packages', [])
    if apt_packages:
        image = image.apt_install(*apt_packages)
    
    pip_packages = image_config.get('pip_packages', [])
    if pip_packages:
        image = image.pip_install(*pip_packages)
    
    setup_commands = task_config.get('setup', [])
    for cmd in setup_commands:
        image = image.run_commands(cmd)
    
    return image


def create_modal_secrets(secret_names: List[str]) -> List[modal.Secret]:
    """Create list of Modal secrets."""
    secrets = []
    for name in secret_names:
        try:
            secret = modal.Secret.from_name(name)
            secrets.append(secret)
        except Exception as e:
            print(f'Warning: Could not load secret {name}: {e}')
    return secrets


def create_file_mounts(file_mounts: Dict[str, Any]) -> Dict[str, modal.Volume]:
    """Create Modal volumes for file mounts.
    
    Translates SkyPilot file_mounts to Modal Volumes.
    Note: This is a simplified implementation that creates or references
    Modal volumes based on the mount configuration.
    """
    volumes = {}
    for mount_path, mount_config in file_mounts.items():
        if isinstance(mount_config, dict):
            volume_name = mount_config.get('name', f'skypilot-{mount_path.replace("/", "-")}')
            persistent = mount_config.get('persistent', True)
            
            try:
                volume = modal.Volume.from_name(volume_name, create_if_missing=True)
                volumes[mount_path] = volume
                print(f'Mounted volume {volume_name} at {mount_path}')
            except Exception as e:
                print(f'Warning: Could not create volume for {mount_path}: {e}')
    
    return volumes


def main():
    parser = argparse.ArgumentParser(description='Execute SkyPilot task on Modal')
    parser.add_argument('config', help='Path to task configuration YAML file')
    args = parser.parse_args()
    
    task_config = load_config(args.config)
    backend_config = load_backend_config()
    
    gpu_spec = task_config.get('gpu', backend_config.get('gpu', 'A100:1'))
    secrets = task_config.get('secrets', backend_config.get('secrets', []))
    env_vars = task_config.get('env', {})
    run_commands = task_config.get('run', [])
    workdir = task_config.get('workdir', '/workspace')
    file_mounts = task_config.get('file_mounts', {})
    timeout = task_config.get('timeout', backend_config.get('timeout', 86400))
    
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
        
        for key, value in env_vars.items():
            if value is not None:
                os.environ[key] = str(value)
        
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
