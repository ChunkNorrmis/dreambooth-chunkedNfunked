import os, sys, torch

def prune_checkpoint(checkpoint, float32=False, token_classes=None):
    if int(checkpoint['global_step']) > 0:
        print(f"This is global step {checkpoint['global_step']}.")
        print('Removing optimizer states from checkpoint')
        pruned_checkpoint = {k: v for k, v in checkpoint.items() if k not in ['state_dict', 'optimizer_states', 'pytorch-lightning_version']}
        pruned_checkpoint['state_dict'] = {k: v.contiguous() if float32 else v.contiguous().half() for k, v in checkpoint['state_dict'].items()}
        pruned_checkpoint['model_precision'] = 'fp32' if float32 else 'fp16'
        pruned_checkpoint['trained_tokens'] = [f"{token_classes[0][0]}_{token_classes[0][1]}", f"{token_classes[1][0]}_{token_classes[1][1]}"]
        print(f"Checkpoint Keys: {pruned_checkpoint.keys()}")
        return pruned_checkpoint
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
