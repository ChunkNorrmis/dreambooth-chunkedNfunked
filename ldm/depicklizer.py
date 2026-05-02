import os, torch, diffusers
from safetensors.torch import save_file, save, load

def depicklize(dict_pickle, out_path=None):
    device = 'cpu'
    suspicious_pickle = torch.load(dict_pickle, map_location=torch.device(device), weights_only=False)
    hefty_pickle = {k: v.contiguous() for k, v in suspicious_pickle['state_dict'].items()}
    metadata = {k: f"{v}" for k, v in suspicious_pickle.items() if k != 'state_dict'}
    metadata['format'] = 'pt'
    if out_path is None:
        nil_pickle = os.path.splitext(dict_pickle)[0] + '.safetensors'
    elif not os.path.isdir(out_path):
        out_path = os.path.dirname(out_path)
    nil_pickle = os.path.join(out_path, dict_pickle.replace('.ckpt', '.safetensors'))
    savetensors = save(hefty_pickle)
    dict_pickless = load(savetensors, device=device)
    for k in hefty_pickle.keys():
        pickless = dict_pickless[k]
        sus_pickle = hefty_pickle[k]
        if not torch.equal(pickless, sus_pickle):
            raise RuntimeError('keys do not match')
    save_file(hefty_pickle, nil_pickle, metadata=metadata)
