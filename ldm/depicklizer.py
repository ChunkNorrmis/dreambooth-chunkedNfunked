import os, sys, torch, shutil
import safetensors.torch as safetorch


def depicklize(dict_pickle, nil_pickle=None):
    suspicious_pickle = torch.load(dict_pickle, map_location=torch.device('cpu'), weights_only=False)
    sus_dict = {k: v.contiguous() for k, v in suspicious_pickle['state_dict'].items()}
    saved = safetorch.save(sus_dict)
    loaded = safetorch.load(saved)
    if equal_tensors(suspicious_pickle['state_dict'], loaded):
        del suspicious_pickle['state_dict']
        metadata = {k: f"{v}" for k, v in suspicious_pickle.items()}
        metadata['format'] = 'pt'
        if nil_pickle is None:
            nil_pickle = dict_pickle.replace('.ckpt', '.safetensors')
        elif os.path.isdir(nil_pickle):
            nil_pickle = os.path.join(nil_pickle, os.path.basename(dict_pickle).replace('.ckpt', '.safetensors'))
        elif not nil_pickle.endswith('.safetensors'):
            nil_pickle = os.path.splitext(nil_pickle)[0] + '.safetensors'
        safetorch.save_file(sus_dict, nil_pickle, metadata=metadata)
    else:
        print('!! Failed -- key mismatch')
        print('!! Aborting safetensors conversion...')
        print(' ')

def equal_tensors(pickle_dict, safe_dict):
    for k in pickle_dict.keys():
        if not torch.equal(pickle_dict[k], safe_dict[k]):
            return False
    return True


if __name__ == '__main__':
    dict_pickle = sys.argv[1]
    nil_pickle = sys.argv[2]
    depicklize(dict_pickle, nil_pickle=nil_pickle)

