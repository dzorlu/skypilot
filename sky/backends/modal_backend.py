"""Modal backend for SkyPilot."""
import json
import os
import subprocess
import tempfile
import typing
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml

from sky import backends
from sky import global_user_state
from sky import sky_logging
from sky.backends import backend_utils
from sky.data import storage as storage_lib
from sky.utils import rich_utils
from sky.utils import ux_utils

if typing.TYPE_CHECKING:
    from sky import resources
    from sky import task as task_lib

PathStr = str

logger = sky_logging.init_logger(__name__)

_MODAL_HANDLE_PREFIX = 'skymodal-'


class ModalResourceHandle(str, backends.ResourceHandle):
    """The name of the Modal deployment prefixed with the handle prefix."""

    def __new__(cls, s, **kw):
        if s.startswith(_MODAL_HANDLE_PREFIX):
            prefixed_str = s
        else:
            prefixed_str = _MODAL_HANDLE_PREFIX + s
        return str.__new__(cls, prefixed_str, **kw)

    def get_cluster_name(self):
        return self.lstrip(_MODAL_HANDLE_PREFIX)


class ModalBackend(backends.Backend['ModalResourceHandle']):
    """Modal backend for running tasks on Modal's serverless infrastructure.
    
    This backend allows SkyPilot tasks to be executed on Modal instead of
    traditional cloud VMs. It handles:
    - Reading Modal configuration (GPU specs, secrets) from config files
    - Creating Modal app deployments with specified resources
    - Executing task commands via Modal functions
    - Syncing workdir and file mounts
    
    Configuration is read from modal_config.yaml in the following locations:
    1. Path specified by MODAL_CONFIG_PATH environment variable
    2. ~/.sky/modal_config.yaml
    3. Current directory ./modal_config.yaml
    
    Example modal_config.yaml:
        gpu: "A100-80GB:8"
        secrets:
          - wandb-secret
          - gcp-credentials
    """

    NAME = 'modal'

    ResourceHandle = ModalResourceHandle

    def __init__(self):
        """Initialize Modal backend."""
        self._config = self._load_config()
        self._validate_config()
        self._deployments: Dict[ModalResourceHandle, Dict[str, Any]] = {}

    def _load_config(self) -> Dict[str, Any]:
        """Load Modal configuration from modal.yaml in backend folder."""
        backend_config_path = Path(__file__).parent / 'modal.yaml'
        
        if backend_config_path.exists():
            with backend_config_path.open('r') as f:
                config = yaml.safe_load(f) or {}
            logger.info(f'Loaded Modal config from {backend_config_path}')
            return config
        
        candidate_paths = [
            Path.home() / '.sky' / 'modal.yaml',
            Path('modal.yaml'),
        ]
        env_path = os.environ.get('MODAL_CONFIG_PATH')
        if env_path:
            candidate_paths.insert(0, Path(env_path))
        
        for path in candidate_paths:
            if path.exists():
                with path.open('r') as f:
                    config = yaml.safe_load(f) or {}
                logger.info(f'Loaded Modal config from {path}')
                return config

        logger.warning(
            'No Modal config file found. Using defaults.')
        return {'gpu': 'A100:1', 'secrets': []}

    def _validate_config(self) -> None:
        """Validate Modal configuration."""
        if 'gpu' not in self._config or not isinstance(self._config['gpu'], str):
            raise ValueError(
                'Modal config must contain "gpu: <TYPE:COUNT>" specification')
        
        if 'secrets' in self._config and not isinstance(
                self._config.get('secrets', []), list):
            raise ValueError(
                'Modal config "secrets" must be a list of secret names')

    def _get_gpu_spec(self) -> str:
        """Get GPU specification from config."""
        return self._config['gpu']

    def _get_secrets(self) -> List[str]:
        """Get list of Modal secret names from config."""
        return self._config.get('secrets', [])

    def check_resources_fit_cluster(
            self, handle: ModalResourceHandle,
            task: 'task_lib.Task') -> Optional['resources.Resources']:
        """Check whether resources of the task are satisfied.
        
        Modal handles resource allocation dynamically, so we just verify
        the request is valid.
        """
        return None

    def _provision(
        self,
        task: 'task_lib.Task',
        to_provision: Optional['resources.Resources'],
        dryrun: bool,
        stream_logs: bool,
        cluster_name: str,
        retry_until_up: bool = False,
        skip_unnecessary_provisioning: bool = False,
    ) -> Tuple[Optional[ModalResourceHandle], bool]:
        """Provision Modal deployment for the task.
        
        This creates the Modal app specification but doesn't deploy yet.
        Actual deployment happens during _execute.
        """
        del to_provision, stream_logs
        
        if dryrun:
            logger.info('Dryrun mode: would provision Modal deployment '
                       f'for cluster {cluster_name}')
            return None, False

        if retry_until_up:
            logger.warning(
                f'retry_until_up is not supported in backend: {self.NAME}. '
                'Ignored the flag.')

        if skip_unnecessary_provisioning:
            logger.warning(
                f'skip_unnecessary_provisioning is not supported in '
                f'backend: {self.NAME}. Ignored the flag.')

        handle = ModalResourceHandle(cluster_name)
        
        self._deployments[handle] = {
            'task_name': task.name,
            'gpu_spec': self._get_gpu_spec(),
            'secrets': self._get_secrets(),
            'created_at': backend_utils.get_timestamp_from_run_timestamp(''),
        }

        logger.info(f'Modal deployment {cluster_name} prepared with GPU: '
                   f'{self._get_gpu_spec()}')

        global_user_state.add_or_update_cluster(
            cluster_name,
            cluster_handle=handle,
            requested_resources=set(task.resources),
            ready=False)

        return handle, False

    def _sync_workdir(self, handle: ModalResourceHandle,
                      workdir: Union[PathStr, Dict[str, Any]],
                      envs_and_secrets: Dict[str, str]) -> None:
        """Sync workdir to Modal.
        
        Modal handles workdir via image building and volume mounts.
        The actual syncing happens when building the Modal image.
        """
        logger.info('Workdir will be synced to Modal during image build.')
        if handle in self._deployments:
            self._deployments[handle]['workdir'] = workdir
            self._deployments[handle]['envs_and_secrets'] = envs_and_secrets

    def _download_file(self, handle: ModalResourceHandle,
                       local_file_path: str, remote_file_path: str) -> None:
        """Download file from Modal deployment.
        
        This is not fully supported as Modal functions are ephemeral.
        Users should use storage mounts for persistent data.
        """
        raise NotImplementedError(
            f'Downloading files from Modal deployments is not supported. '
            f'Use storage mounts or Modal volumes for persistent data.')

    def _sync_file_mounts(
        self,
        handle: ModalResourceHandle,
        all_file_mounts: Optional[Dict[PathStr, PathStr]],
        storage_mounts: Optional[Dict[PathStr, storage_lib.Storage]],
    ) -> None:
        """Sync file mounts to Modal.
        
        File mounts are handled by baking them into the Modal image
        or using Modal volumes.
        """
        if storage_mounts:
            logger.warning(
                'Storage mounts are not fully supported with Modal backend. '
                'Files will be baked into the image instead.')
        
        if handle in self._deployments:
            self._deployments[handle]['file_mounts'] = all_file_mounts or {}

        logger.info('File mounts will be handled during Modal image build.')

    def _setup(self, handle: ModalResourceHandle, task: 'task_lib.Task',
               detach_setup: bool) -> None:
        """Setup phase for Modal deployment.
        
        This prepares the Modal function but doesn't execute it yet.
        The setup commands from the task will be incorporated into the
        Modal image build.
        """
        del detach_setup

        logger.info(f'Setting up Modal deployment for {handle}')

        if handle in self._deployments:
            self._deployments[handle]['setup_commands'] = task.setup
            self._deployments[handle]['task'] = task

        global_user_state.add_or_update_cluster(
            handle.get_cluster_name(),
            cluster_handle=handle,
            requested_resources=set(task.resources),
            ready=True)

        logger.info(f'Modal deployment {handle} ready for execution')

    def _execute(self,
                 handle: ModalResourceHandle,
                 task: 'task_lib.Task',
                 dryrun: bool = False) -> Optional[int]:
        """Execute the task on Modal.
        
        This creates a Modal Python script that defines the app and function,
        then invokes it via the Modal CLI.
        """
        if task.num_nodes > 1:
            logger.warning(
                'Multi-node tasks are mapped to multi-GPU single-node '
                'execution on Modal.')

        if task.run is None:
            logger.info(f'Nothing to run; run command not specified:\n{task}')
            return None

        if dryrun:
            logger.info(f'Dryrun complete. Would have run:\n{task}')
            return None

        return self._execute_task_on_modal(handle, task)

    def _execute_task_on_modal(self, handle: ModalResourceHandle,
                                task: 'task_lib.Task') -> Optional[int]:
        """Execute task by invoking modal.py script with task configuration."""
        if callable(task.run):
            raise NotImplementedError(
                'Tasks with callable run commands are not supported in '
                'ModalBackend.')

        deployment_info = self._deployments.get(handle, {})
        gpu_spec = deployment_info.get('gpu_spec', self._get_gpu_spec())
        secrets = deployment_info.get('secrets', self._get_secrets())
        workdir = deployment_info.get('workdir')
        envs = deployment_info.get('envs_and_secrets', {})
        file_mounts = deployment_info.get('file_mounts', {})

        task_config = self._create_task_config(
            task=task,
            gpu_spec=gpu_spec,
            secrets=secrets,
            workdir=workdir,
            envs=envs,
            file_mounts=file_mounts,
        )

        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(task_config, f)
            config_path = f.name

        try:
            logger.info(f'Executing task on Modal: {handle}')
            
            modal_script_path = Path(__file__).parent / 'modal.py'
            cmd = ['python3', str(modal_script_path), config_path]
            
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=False,
                text=True,
            )
            
            if result.returncode != 0:
                logger.error(f'Modal execution failed with code {result.returncode}')
            
            return result.returncode

        finally:
            os.unlink(config_path)

    def _create_task_config(
        self,
        task: 'task_lib.Task',
        gpu_spec: str,
        secrets: List[str],
        workdir: Optional[Union[PathStr, Dict[str, Any]]],
        envs: Dict[str, str],
        file_mounts: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create task configuration for modal.py."""
        env_dict = {**envs}
        if task.envs:
            env_dict.update(task.envs)

        setup_commands = []
        if task.setup:
            if isinstance(task.setup, str):
                setup_commands = [task.setup]
            elif isinstance(task.setup, list):
                setup_commands = task.setup

        if isinstance(task.run, str):
            run_commands = [task.run]
        elif isinstance(task.run, list):
            run_commands = task.run
        else:
            run_commands = [str(task.run)] if task.run else []

        config = {
            'gpu': gpu_spec,
            'secrets': secrets,
            'env': env_dict,
            'setup': setup_commands,
            'run': run_commands,
            'file_mounts': file_mounts,
            'workdir': str(workdir) if workdir else '/workspace',
        }

        return config

    def _post_execute(self, handle: ModalResourceHandle, down: bool) -> None:
        """Post-execution cleanup."""
        logger.info(f'Modal task execution completed for {handle}')
        if down:
            logger.info(f'Tearing down Modal deployment {handle}')
            self._teardown(handle, terminate=True)

    def _teardown(self,
                  handle: ModalResourceHandle,
                  terminate: bool,
                  purge: bool = False):
        """Teardown Modal deployment."""
        del purge

        if not terminate:
            logger.warning(
                'ModalBackend.teardown() will terminate deployments, '
                'despite receiving terminate=False.')

        if handle in self._deployments:
            del self._deployments[handle]

        cluster_name = handle.get_cluster_name()
        global_user_state.remove_cluster(cluster_name, terminate=True)
        
        logger.info(f'Modal deployment {handle} torn down')

    def add_storage_objects(self, task: 'task_lib.Task') -> None:
        """Add storage objects for the task."""
        pass

    def _teardown_ephemeral_storage(self, task: 'task_lib.Task') -> None:
        """Teardown ephemeral storage."""
        pass
