import torch
import random
import numpy as np


import sys
sys.path.append("..")
from data.cityscapes_dataset import CityscapesDataset

def activate_gpus(config):
    """Identify which GPUs to activate
        Args:
            config: Configuration dictionary with project hyperparameters
        Returns:
            dict: Required information for GPU/CPU training
    """
    str_ids = config['gpu_ids'].split(',')
    gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            gpu_ids.append(id)

    if torch.cuda.is_available() and len(gpu_ids) > 0:
        use_gpu = True
    else:
        use_gpu = False
        gpu_ids = []
    device = torch.device("cuda:" + str(gpu_ids[0]) if use_gpu else "cpu")
    return {'device': device, 'gpu_ids': gpu_ids, 'use_gpu': use_gpu}

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def get_dataloader(dataset_args, dataloader_args):
    dataset = CityscapesDataset(**dataset_args)
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_args)
    return dataloader
    