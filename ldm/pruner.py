from dreambooth_helpers.joepenna_dreambooth_config import JoePennaDreamboothConfigSchemaV1

def prune_checkpoint(old_state):
    conf = JoePennaDreamboothConfigSchemaV1()
    if int(old_state['global_step']) > 0:
        print(f"This is global step {old_state['global_step']}.")
        print(f"Checkpoint Keys: {old_state.keys()}")
        print('Removing optimizer states from checkpoint')
        pruned_checkpoint = {k: v for k, v in old_state.items() if k != "optimizer_states" and k != 'state_dict'}
        if conf.model_precision():
            pruned_checkpoint['state_dict'] = {k: v.contiguous() for k, v in old_state['state_dict'].items()}
            pruned_checkpoint['precision'] = 'float32'
        else:
            pruned_checkpoint['state_dict'] = {k: v.half().contiguous() for k, v in old_state['state_dict'].items()}
            pruned_checkpoint['precision'] = 'float16'
        return pruned_checkpoint
