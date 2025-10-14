"""Modal Cloud.

This provides a Cloud-level integration so users can select `--cloud modal`.
Under the hood, execution is routed to the Modal execution path (serverless
functions), not Ray-based VMs.

Scope: implement required Cloud interfaces with sensible placeholders and
unsupported feature flags reflecting Modal's capabilities (single-node,
no storage mounting, etc.). Pricing/instance constructs are not applicable
and return default values.
"""
from typing import Dict, Iterator, List, Optional, Tuple, Union

from pathlib import Path
import os
import yaml
from sky import clouds
from sky.utils import registry
from sky.utils import resources_utils


@registry.CLOUD_REGISTRY.register
class Modal(clouds.Cloud):
    """Modal serverless GPU cloud.

    A thin shim to expose Modal as a SkyPilot cloud, while the execution is
    handled by the Modal runtime path.
    """

    _REPR = 'Modal'

    # Capabilities and limitations
    _CLOUD_UNSUPPORTED_FEATURES = {
        clouds.CloudImplementationFeatures.STOP: 'Stopping not supported.',
        clouds.CloudImplementationFeatures.MULTI_NODE:
            'Multi-node not supported; single node with multiple GPUs only.',
        clouds.CloudImplementationFeatures.CUSTOM_DISK_TIER:
            'Custom disk tier is not applicable.',
        clouds.CloudImplementationFeatures.CUSTOM_NETWORK_TIER:
            'Custom network tier is not applicable.',
        clouds.CloudImplementationFeatures.STORAGE_MOUNTING:
            'Object store mounting is not supported on Modal.',
        clouds.CloudImplementationFeatures.HIGH_AVAILABILITY_CONTROLLERS:
            'Controllers are not supported on Modal.',
        clouds.CloudImplementationFeatures.CUSTOM_MULTI_NETWORK:
            'Multiple network interfaces not supported.',
    }

    PROVISIONER_VERSION = clouds.ProvisionerVersion.SKYPILOT
    STATUS_VERSION = clouds.StatusVersion.SKYPILOT
    OPEN_PORTS_VERSION = clouds.OpenPortsVersion.UPDATABLE

    # Config discovery (mirrors previous backend behavior)
    @staticmethod
    def load_config() -> Dict[str, any]:
        candidates = [
            Path(__file__).parent / 'modal.yaml',             # repo default
            Path.home() / '.sky' / 'modal.yaml',              # user override
            Path('modal.yaml'),                               # cwd override
        ]
        env_path = os.environ.get('MODAL_CONFIG_PATH')
        if env_path:
            candidates.insert(0, Path(env_path))
        for p in candidates:
            if p.exists():
                with p.open('r') as f:
                    return yaml.safe_load(f) or {}
        return {}

    @classmethod
    def _unsupported_features_for_resources(
        cls, resources: 'clouds.resources_lib.Resources'
    ) -> Dict[clouds.CloudImplementationFeatures, str]:
        del resources
        return cls._CLOUD_UNSUPPORTED_FEATURES

    @classmethod
    def _max_cluster_name_length(cls) -> Optional[int]:
        # Follow generous defaults
        return None

    # Regions/Zones: Modal abstracts location. Present a single logical region.
    @classmethod
    def regions_with_offering(
        cls,
        instance_type: str,
        accelerators: Optional[Dict[str, int]],
        use_spot: bool,
        region: Optional[str],
        zone: Optional[str],
    ) -> List[clouds.Region]:
        del instance_type, accelerators, use_spot, zone
        regions = [clouds.Region('global').set_zones([])]
        if region is not None:
            regions = [r for r in regions if r.name == region]
        return regions

    @classmethod
    def zones_provision_loop(
        cls,
        *,
        region: str,
        num_nodes: int,
        instance_type: str,
        accelerators: Optional[Dict[str, int]] = None,
        use_spot: bool = False,
    ) -> Iterator[Optional[List['clouds.Zone']]]:
        del region, num_nodes, instance_type, accelerators, use_spot
        # No zones; yield None once to indicate region-level provision.
        yield None

    @classmethod
    def get_zone_shell_cmd(cls) -> Optional[str]:
        return None

    def instance_type_to_hourly_cost(
        self,
        instance_type: str,
        use_spot: bool,
        region: Optional[str] = None,
        zone: Optional[str] = None,
    ) -> float:
        del instance_type, use_spot, region, zone
        return 0.0

    def accelerators_to_hourly_cost(
        self,
        accelerators: Dict[str, int],
        use_spot: bool,
        region: Optional[str] = None,
        zone: Optional[str] = None,
    ) -> float:
        del accelerators, use_spot, region, zone
        return 0.0

    def get_egress_cost(self, num_gigabytes: float) -> float:
        del num_gigabytes
        return 0.0

    @classmethod
    def get_default_instance_type(
        cls,
        cpus: Optional[str] = None,
        memory: Optional[str] = None,
        disk_tier: Optional[resources_utils.DiskTier] = None,
        region: Optional[str] = None,
        zone: Optional[str] = None,
    ) -> Optional[str]:
        del cpus, memory, disk_tier, region, zone
        # Not applicable; instance types are not a concept for Modal tasks.
        return None

    @classmethod
    def get_accelerators_from_instance_type(
        cls, instance_type: str
    ) -> Optional[Dict[str, Union[int, float]]]:
        del instance_type
        return None

    @classmethod
    def get_vcpus_mem_from_instance_type(
        cls, instance_type: str
    ) -> Tuple[Optional[float], Optional[float]]:
        del instance_type
        return None, None

    @classmethod
    def get_image_size(cls, image_id: str, region: Optional[str]) -> float:
        del image_id, region
        return 0.0

    def instance_type_exists(self, instance_type: str) -> bool:
        del instance_type
        return True

    def validate_region_zone(
        self, region: Optional[str], zone: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        del zone
        # Accept only the logical 'global' region or None
        if region is None or region == 'global':
            return region, None
        raise ValueError('Modal only supports region "global" or unspecified.')

    def make_deploy_resources_variables(
        self,
        resources: 'clouds.resources_lib.Resources',
        cluster_name: resources_utils.ClusterName,
        region: 'clouds.Region',
        zones: Optional[List['clouds.Zone']],
        num_nodes: int,
        dryrun: bool = False,
        volume_mounts: Optional[List['clouds.volume_lib.VolumeMount']] = None,
    ) -> Dict[str, Optional[Union[str, bool]]]:
        # Not used by the execution path (handled in Modal runtime). Provide a
        # minimal dict for compatibility if referenced.
        del resources, cluster_name, region, zones, num_nodes, dryrun, volume_mounts
        return {}

    @classmethod
    def _check_compute_credentials(
        cls,
    ) -> Tuple[bool, Optional[Union[str, Dict[str, str]]]]:  # type: ignore[name-defined]
        # Assume credentials are managed by Modal CLI and secrets; don't block.
        return True, None

    def get_credential_file_mounts(self) -> Dict[str, str]:
        # Credentials are passed via Modal secrets.
        return {}


