import os, torch, diffusers
from safetensors.torch import save_file, load_file

def depicklize(dict_pickle, out_path=None, only_weights=False):
    device = 'cpu'
    suspicious_pickle = torch.load(dict_pickle, map_location=device, weights_only=only_weights)
        if not only_weights:
            metadata = {k: f"{v}" for k, v in suspicious_pickle.items() if k != 'state_dict'}
            metadata['format'] = 'ckpt'
        else:
            metadata = {'format': 'ckpt'}
    hefty_pickle = {k: v.contiguous() for k, v in suspicious_pickle['state_dict'].items()}
    if out_path is None:
        out_path = dict_pickle
    pickle_path, pickle = os.path.split(out_path)
    nil_pickle = pickle.replace(f".{metadata['format']}", '.safetensors')
    nil_pickle = os.path.join(pickle_path, nil_pickle)
    save_file(hefty_pickle, nil_pickle, metadata=metadata)
    dict_picholas = load_file(nil_pickle, device=device)
    for k in hefty_pickle.keys():
        pickeless = dict_picholas[k]
        sus_pickle = hefty_pickle[k]
        if not torch.equal(pickeless, sus_pickle):
            raise RuntimeError('keys do not match')
    return nil_pickle
