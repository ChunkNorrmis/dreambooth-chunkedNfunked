import os, torch
from ldm.depicklizer import depicklize


def get_last_ckpt(steps, ckpt, dst):
    if not os.path.exists(ckpt):
        return
    device = torch.device('cpu')
    trained_steps = int(torch.load(ckpt, map_location=device, weights_only=False)['global_step'])
    if steps != trained_steps:
        return
    else: depicklize(ckpt, nil_pickle=dst)
    
