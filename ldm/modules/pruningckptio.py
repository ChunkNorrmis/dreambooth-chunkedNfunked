import os
from pytorch_lightning.plugins.io.torch_plugin import TorchCheckpointIO

from typing import Any, Callable, Dict, Optional
from pytorch_lightning.utilities.types import _PATH
from ldm.pruner import prune_checkpoint, prune_pickle
from safetensors.torch import save_file
 
class PruningCheckpointIO(TorchCheckpointIO):
    def __init__(self, precision='float16', safetensors=False):
        self.precision = precision
        self.safetensors = safetensors
     
    def save_checkpoint(self, checkpoint, path, storage_options=None):
        if self.safetensors:
            pruned_pickle, metadata = prune_pickle(checkpoint, precision=self.precision)
            pickle_path, pickle = os.path.split(path)
            pickle = pickle.replace('.ckpt', '.safetensors')
            nil_pickle = os.path.join(pickle_path, pickle)
            save_file(pruned_pickle, nil_pickle, metadata=metadata)
        else:
            pruned_checkpoint = prune_checkpoint(checkpoint, precision=self.precision)
            TorchCheckpointIO.save_checkpoint(self, pruned_checkpoint, path, storage_options)


