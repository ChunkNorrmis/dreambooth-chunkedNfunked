import os
from pytorch_lightning.plugins.io.torch_plugin import TorchCheckpointIO

from typing import Any, Callable, Dict, Optional
from pytorch_lightning.utilities.types import _PATH
from ldm.pruner import prune_checkpoint
 
class PruningCheckpointIO(TorchCheckpointIO):
    def __init__(self, dtype='float16'):
        self.dtype = dtype
    
 def save_checkpoint(self, checkpoint, path, storage_options=None):
        pruned_checkpoint = prune_checkpoint(checkpoint, dtype=self.dtype)
        TorchCheckpointIO.save_checkpoint(self, pruned_checkpoint, path, storage_options)

