def prune_checkpoint(checkpoint, dtype='float16'):
    if int(checkpoint['global_step']) > 0:
        print(f"This is global step {checkpoint['global_step']}.")
        print('Removing optimizer states from checkpoint')
        pruned_checkpoint = {k: v for k, v in checkpoint.items() if k != "optimizer_states" and k != 'state_dict'}
        pruned_checkpoint['dtype'] = dtype
        if dtype == 'float16':     pruned_checkpoint['state_dict'] = {k: v.half().contiguous() for k, v in checkpoint['state_dict'].items()}
        else:
            pruned_checkpoint['state_dict'] = {k: v.half().contiguous() for k, v in checkpoint['state_dict'].items()}

      
        print(f"Checkpoint Keys: {pruned_checkpoint.keys()}")
        return pruned_checkpoint

def prune_pickle(checkpoint, dtype='float16'):
    if int(checkpoint['global_step']) > 0:
        print(f"This is global step {checkpoint['global_step']}.")
        print('Removing optimizer states from checkpoint')
        metadata = {k: f"{v}" for k, v in checkpoint.items() if k != 'optimizer_states' and k != 'state_dict'}
        metadata['format'] = 'pt'
        metadata['dtype'] = dtype
        if dtype == 'float16':
            nil_pickle = {k: v.half().contiguous() for k, v in checkpoint['state_dict'].items()}
        else:
            nil_pickle = {k: v.half().contiguous() for k, v in checkpoint['state_dict'].items()}
        keys = [k for k in metadata.keys()]
        keys += 'state_dict'
        
        print(f"Checkpoint Keys: {keys}")
        return nil_pickle, metadata
