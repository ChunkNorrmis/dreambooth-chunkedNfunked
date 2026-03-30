def prune_checkpoint(old_state):
    if int(old_state['global_step']) > 0:
        print(f"This is global step {old_state['global_step']}.")
        print('Removing optimizer states from checkpoint')
        pruned_checkpoint = {k: v for k, v in old_state.items() if k != "optimizer_states" and k != 'state_dict'}
        pruned_checkpoint['state_dict'] = {k: v.contiguous() for k, v in old_state['state_dict'].items()}
        pruned_checkpoint['precision'] = 'float32'
        #pruned_checkpoint['state_dict'] = {k: v.half().contiguous() for k, v in old_state['state_dict'].items()}
        #pruned_checkpoint['precision'] = 'float16'
        print(f"Checkpoint Keys: {pruned_checkpoint.keys()}")
        return pruned_checkpoint
