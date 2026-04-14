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
    def save_checkpoint(
            self, 
            checkpoint: Dict[str, Any], 
            path: _PATH, 
            storage_options: Optional[Any] = None
        ) -> None:
        pruned_checkpoint = prune_checkpoint(checkpoint, precision=self.precision)
        if not self.sftsr:
            TorchCheckpointIO.save_checkpoint(self, pruned_checkpoint, path, storage_options)
        else:
            if path.endswith('.ckpt'):
                path = path.replace('.ckpt', '.safetensors')
            elif os.path.isdir(path):
                path = os.path.join(path, 'last.safetensors')
            else: 
                path = f"{path}.safetensors"
            save_file(pruned_checkpoint, path)
