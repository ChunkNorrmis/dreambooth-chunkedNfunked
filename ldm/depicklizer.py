import os, torch, diffusers, sys
from safetensors.torch import save_file, save

def depicklize(dict_pickle, nil_pickle=None):
    device = 'cpu'
    suspicious_pickle = torch.load(dict_pickle, map_location=torch.device(device), weights_only=False)
    hefty_pickle = {k: v.contiguous() for k, v in suspicious_pickle['state_dict'].items()}
    metadata = {k: f"{v}" for k, v in suspicious_pickle.items() if k != 'state_dict'}
    metadata['format'] = 'pt'
    if nil_pickle is None:
        nil_pickle = os.path.splitext(dict_pickle)[0] + '.safetensors'
    elif os.path.isdir(nil_pickle):
        nil_pickle = os.path.join(nil_pickle, dict_pickle.replace('.ckpt', '.safetensors'))
    dict_pickless = save(hefty_pickle)
    for k in hefty_pickle.keys():
        pickless = dict_pickless[k]
        sus_pickle = hefty_pickle[k]
        if not torch.equal(pickless, sus_pickle):
            raise RuntimeError('keys do not match')
    save_file(hefty_pickle, nil_pickle, metadata=metadata)
