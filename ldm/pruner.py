import os, sys, torch

def prune_checkpoint(checkpoint, dtype='float16'):
    if int(checkpoint['global_step']) > 0:
        print(f"This is global step {checkpoint['global_step']}.")
        print('Removing optimizer states from checkpoint')
        if 'optimizer_states' in checkpoint.keys():
            del checkpoint['optimizer_states']
        if dtype == 'float16':
            pruned_checkpoint = {k: v for k, v in checkpoint.items() if k != 'state_dict'}
            pruned_checkpoint['state_dict'] = {k: v.half() for k, v in checkpoint['state_dict'].items()}
        else:
            pruned_checkpoint = checkpoint.copy()
        pruned_checkpoint['precision'] = dtype
        print(f"Checkpoint Keys: {pruned_checkpoint.keys()}")
        return pruned_checkpoint
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
