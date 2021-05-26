import argparse

import torch
import yaml


def load_config(config_file):
    config = _load_config_yaml(config_file)
    # Get a device to train on
    device_str = config.get('device', None)
    if device_str is not None:
        print(f"Device specified in config: '{device_str}'")
    #     if device_str.startswith('cuda') and not torch.cuda.is_available():
    #         logger.warn('CUDA not available, using CPU')
    #         device_str = 'cpu'
    else:
        device_str = 'cpu'
        print(f"Using '{device_str}' device")

    device = torch.device(device_str)
    config['device'] = device
    return config


def _load_config_yaml(config_file):
    return yaml.safe_load(open(config_file, 'r'))
