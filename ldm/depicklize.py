import os, torch, diffusers
from safetensors.torch import save_file, load_file

def depicklize(suspicious_pickle, device=None, only_weights=False):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if suspicious_pickle.endswith('.ckpt'):
        dict_pickle = torch.load(suspicious_pickle, map_location=device, weights_only=only_weights)
        if not only_weights:
            metadata = {k: f"{v}" for k, v in dict_pickle.items() if k != 'state_dict'}
            metadata['format'] = 'ckpt'
        else:
            metadata = {'format': 'ckpt'}
    pickle_dict = {k: v.contiguous() for k, v in dict_pickle['state_dict'].items()}
    nil_pickle = os.path.basename(suspicious_pickle).replace(f".{metadata['format']}", '.safetensors')
    nil_pickle = os.path.join(os.path.dirname(suspicious_pickle), nil_pickle)
    save_file(pickle_dict, nil_pickle, metadata=metadata)
    loaded = load_file(nil_pickle, device=device)
    for k in pickle_dict.keys():
        if not torch.equal(loaded[k], pickle_dict[k]):
            raise RuntimeError('keys do not match')
        else:
            print('conversion successful')
