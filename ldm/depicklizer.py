import os, sys, torch
from safetensors.torch import save, load, save_file

def depicklize(dict_pickle, nil_pickle=None):
    print('Depicklizing suspicious pickle(s)...')
    suspicious_pickle = torch.load(dict_pickle, map_location=torch.device('cpu'), weights_only=False)
    metadata = {k: f"{v}" for k, v in suspicious_pickle.items() if k != 'state_dict'}
    metadata['format'] = 'pt'
    hefty_pickle = {k: v.contiguous() for k, v in suspicious_pickle['state_dict'].items()}
    savedtensors = save(hefty_pickle)
    dict_pickless = load(savedtensors)
    for key in suspicious_pickle['state_dict'].keys():
        if not torch.equal(dict_pickless[key], suspicious_pickle['state_dict'][key]):
            raise RuntimeError('keys do not match')
    if nil_pickle is None:
        nil_pickle = dict_pickle.replace('.ckpt', '.safetensors')
    elif os.path.isdir(nil_pickle):
        dict_pickle = dict_pickle.replace('.ckpt', '.safetensors')
        nil_pickle = os.path.join(nil_pickle, dict_pickle)
    save_file(hefty_pickle, nil_pickle, metadata=metadata)


