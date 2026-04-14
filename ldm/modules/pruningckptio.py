import os
from pytorch_lightning.plugins.io.torch_plugin import TorchCheckpointIO
from safetensors.torch import save_file

from typing import Any, Callable, Dict, Optional
from pytorch_lightning.utilities.types import _PATH
from ldm.pruner import prune_checkpoint
 
class PruningCheckpointIO(TorchCheckpointIO):
    def __init__(self, precision='float16', sftsr=False):
        self.precision = precision
        self.sftsr = sftsr
     
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        if int(checkpoint['global_step']) > 0:
            print(f"This is global step {checkpoint['global_step']}.")
        print('Removing optimizer states from checkpoint')
        
        if self.sftsr:
            for k, v in checkpoint.items():
                if isinstance(v, torch.Tensor):
                    pruned_checkpoint[k] = v if k != "optimizer_states" and k != 'state_dict'
                elif isinstance(v, int):
                    pruned_checkpoint[k] = torch.tensor(v)
                elif isinstance(v, str):
                    metadata[k] = v
        else:
            pruned_checkpoint = {k: v for k, v in checkpoint.items() if k != "optimizer_states" and k != 'state_dict'}
        
        if precision == 'float16':
            pruned_checkpoint['state_dict'] = {k: v.half().contiguous() for k, v in checkpoint['state_dict'].items()}
        elif precision == 'float32':
            pruned_checkpoint['state_dict'] = {k: v.contiguous() for k, v in checkpoint['state_dict'].items()}

        print(f"Checkpoint Keys: {pruned_checkpoint.keys()}")

        if not self.sftsr:
            pruned_checkpoint['precision'] = self.precision
            TorchCheckpointIO.save_checkpoint(self, pruned_checkpoint, path, storage_options)
        else:
            metadata = {'precision': self.precision}
            if path.endswith('.ckpt'):
                path = path.replace('.ckpt', '.safetensors')
            elif os.path.isdir(path):
                path = os.path.join(path, 'last.safetensors')
            else: 
                path = f"{path}.safetensors"
            save_file(pruned_checkpoint, path, metadata=metadata)


