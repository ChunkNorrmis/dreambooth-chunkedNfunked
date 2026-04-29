def prune_checkpoint(checkpoint, dtype='float16'):
    if int(checkpoint['global_step']) > 0:
        print(f"This is global step {checkpoint['global_step']}.")
        print('Removing optimizer states from checkpoint')
        pruned_checkpoint = {k: checkpoint[k] for k in checkpoint.keys() if k != 'optimizer_states' and k != 'state_dict'}
        pruned_checkpoint['precision'] = dtype
        if dtype == 'float16':
            pruned_checkpoint['state_dict'] = {k: v.half().contiguous() for k, v in checkpoint['state_dict'].items()}
        else:
            pruned_checkpoint['state_dict'] = {k: v.float().contiguous() for k, v in checkpoint['state_dict'].items()}
      
        print(f"Checkpoint Keys: {pruned_checkpoint.keys()}")
        return pruned_checkpoint


def savetensors(fun):
    checkpoint, location, data = fun
    return save_file(checkpoint, location, metadata=data)

@ savetensors
def prune_pickle(checkpoint, dtype='float16', path=None):
    if int(checkpoint['global_step']) > 0:
        print(f"This is global step {checkpoint['global_step']}.")
        print('Removing optimizer states from checkpoint')
        if dtype == 'float16':
            nil_pickle = {k: v.half().contiguous() for k, v in checkpoint['state_dict'].items()}
        else:
            nil_pickle = {k: v.float().contiguous() for k, v in checkpoint['state_dict'].items()}
        metadata = {k: f"{checkpoint[k]}" for k in checkpoint.keys() if k != 'optimizer_states' and k != 'state_dict'}
        metadata['format'] = 'pt'
        metadata['precision'] = dtype
        keys_list = metadata.keys() + 'state_dict'
         
        print(f"Checkpoint Keys: {keys_list}")
        return nil_pickle, path, metadata
