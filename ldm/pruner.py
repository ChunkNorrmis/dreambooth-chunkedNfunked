import os, sys, torch

def prune_checkpoint(checkpoint, float32=False):
    if int(checkpoint['global_step']) > 0:
        print(f"This is global step {checkpoint['global_step']}.")
        print('Removing optimizer states from checkpoint')
        pruned_checkpoint = {k: v for k, v in checkpoint.items() if k != 'state_dict' and k != 'optimizer_states'}
        state_dict = {k: v if float32 else v.half() for k, v in checkpoint['state_dict'].items()}
        pruned_checkpoint['state_dict'] = state_dict
        pruned_checkpoint['precision'] = 'fp32' if float32 else 'fp16'
        print(f"Checkpoint Keys: {pruned_checkpoint.keys()}")
        return pruned_checkpoint
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
