import os, sys, torch

def prune_checkpoint(checkpoint, dtype='float16'):
    if int(checkpoint['global_step']) > 0:
        print(f"This is global step {checkpoint['global_step']}.")
        print('Removing optimizer states from checkpoint')
        pruned_checkpoint = {k: v for k, v in checkpoint.items() if k != 'state_dict' and k != 'optimizer_states'}
        if dtype == 'float16':
            pruned_checkpoint['state_dict'] = {k: v.half() for k, v in checkpoint['state_dict'].items()}
        else:
            pruned_checkpoint['state_dict'] = {k: v for k, v in checkpoint['state_dict'].items()}
        pruned_checkpoint['precision'] = dtype
        print(f"Checkpoint Keys: {pruned_checkpoint.keys()}")
        return pruned_checkpoint
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
