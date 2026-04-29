import os
from pytorch_lightning.plugins.io.torch_plugin import TorchCheckpointIO

from typing import Any, Callable, Dict, Optional
from pytorch_lightning.utilities.types import _PATH
from ldm.pruner import prune_checkpoint, prune_pickle
from safetensors.torch import save_file
 
class PruningCheckpointIO(TorchCheckpointIO):
    def __init__(self, dtype='float16', format='safetensors'):
        self.dtype = dtype
        self.format = format
     
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        if self.format == 'safetensors':
            nil_pickle, path, metadata = prune_pickle(checkpoint, dtype=self.dtype, path=path)
            save_file(nil_pickle, path, metadata=metadata)
        else:
            pruned_checkpoint = prune_checkpoint(checkpoint, dtype=self.dtype)
            TorchCheckpointIO.save_checkpoint(self, pruned_checkpoint, path, storage_options)

