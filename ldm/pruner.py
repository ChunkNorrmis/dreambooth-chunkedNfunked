def prune_checkpoint(checkpoint, precision='float16'):
    if int(checkpoint['global_step']) > 0:
        print(f"This is global step {checkpoint['global_step']}.")
        print('Removing optimizer states from checkpoint')
        pruned_checkpoint = {k: v for k, v in checkpoint.items() if k != "optimizer_states" and k != 'state_dict'}
        if precision == 'float16':
            pruned_checkpoint['state_dict'] = {k: v.half().contiguous() for k, v in checkpoint['state_dict'].items()}
            pruned_checkpoint['precision'] = 'float16'
        elif precision == 'float32':
            pruned_checkpoint['state_dict'] = {k: v.contiguous() for k, v in checkpoint['state_dict'].items()}
            pruned_checkpoint['precision'] = 'float32'
        
        print(f"Checkpoint Keys: {pruned_checkpoint.keys()}")
        return pruned_checkpoint
