import os, sys, torch, shutil
import safetensors.torch as safetorch


def depicklize(dict_pickle, nil_pickle=None):
    def equal_tensors(sus_dict, loaded):
        print('Comparing model keys...')
        for k in sus_dict.keys():
            if not torch.equal(sus_dict[k], loaded[k]):
                print('Fail -- key mismatch')
                print('Aborting safetensors conversion...')
                print(' ')
                return False
        return True

    print(f" Depickling model checkpoints")
    suspicious_pickle = torch.load(dict_pickle, map_location=torch.device('cpu'), weights_only=False)
    sus_dict = {k: v.contiguous() for k, v in suspicious_pickle['state_dict'].items()}
    del suspicious_pickle['state_dict']
    metadata = {k: f"{v}" for k, v in suspicious_pickle.items()}
    metadata['format'] = 'pt'
    if nil_pickle is None:
        nil_pickle = dict_pickle.replace('.ckpt', '.safetensors')
    elif os.path.isdir(nil_pickle):
        nil_pickle = os.path.join(nil_pickle, os.path.basename(dict_pickle).replace('.ckpt', '.safetensors'))
    elif not nil_pickle.endswith('.safetensors'):
        nil_pickle = os.path.splitext(nil_pickle)[0] + '.safetensors'
    saved = safetorch.save(sus_dict)
    loaded = safetorch.load(saved)
    if equal_tensors(sus_dict, loaded):
        safetorch.save_file(sus_dict, nil_pickle, metadata=metadata)
    else:
        nil_pickle = os.path.join('trained_models', os.path.basename(dict_pickle))
        shutil.move(dict_pickle, nil_pickle)


if __name__ == '__main__':
    dict_pickle = sys.argv[1]
    nil_pickle = sys.argv[2]
    depicklize(dict_pickle, nil_pickle=nil_pickle)

