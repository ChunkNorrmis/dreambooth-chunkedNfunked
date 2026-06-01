import os, sys, torch, shutil
import safetensors.torch as safetorch


def depicklize(dict_pickle, nil_pickle=None):
    def equal_tensors(sus_dict, loaded):
        print('Comparing model keys...')
        for k in sus_dict.keys():
            if not torch.equal(sus_dict[k], loaded[k]):
                print('!! Key mismatch error !!')
                print('Aborting safetensors conversion.')
                print(' ')
                return False
        return True

    print(' ')
    print(f"Depicklizing '{os.path.basename(dict_pickle)}'")
    suspicious_pickle = torch.load(dict_pickle, map_location=torch.device('cpu'), weights_only=False)
    sus_dict = {k: v.contiguous() for k, v in suspicious_pickle['state_dict'].items()}
    del suspicious_pickle['state_dict']
    metadata = {k: f"{v}" for k, v in suspicious_pickle.items()}
    metadata['format'] = 'pt'
    if nil_pickle is None:
        nil_pickle = dict_pickle.replace('.ckpt', '.safetensors')
    elif os.path.isdir(nil_pickle):
        nil_pickle = os.path.join(nil_pickle, os.path.basename(dict_pickle).replace('.ckpt', '.safetensors'))
    saved = safetorch.save(sus_dict)
    loaded = safetorch.load(saved)
    if equal_tensors(sus_dict, loaded):
        print('Key check pass...')
        print(f"Saving to './{os.path.relpath(nil_pickle)}'")
        print(' ')
        safetorch.save_file(sus_dict, nil_pickle, metadata=metadata)
    else:
        nil_pickle = os.path.join('trained_models', os.path.basename(dict_pickle))
        print(f"Moving model to './{os.path.relpath(nil_pickle)}'")
        print(' ')
        shutil.move(dict_pickle, nil_pickle)

