import os
from pytorch_lightning.plugins.io.torch_plugin import TorchCheckpointIO

from typing import Any, Callable, Dict, Optional
from pytorch_lightning.utilities.types import _PATH
from ldm.pruner import prune_checkpoint, prune_pickle
import safetensors.torch import save_file, save_model
 
class PruningCheckpointIO(TorchCheckpointIO):
    def __init__(self, precision='float16', format=None):
        self.precision = precision
        self.format = format
     
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        path = f"{os.path.splitext(path)[0]}{self.format}"
        if self.format == '.safetensors':
            nil_pickle, metadata = prune_pickle(checkpoint, precision=self.precision)
            save_model(nil_pickle, path, metadata=metadata)
        else:
            pruned_checkpoint = prune_checkpoint(checkpoint, precision=self.precision)
            TorchCheckpointIO.save_checkpoint(self, pruned_checkpoint, path, storage_options)


