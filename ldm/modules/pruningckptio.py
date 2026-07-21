import os
from pytorch_lightning.plugins.io.torch_plugin import TorchCheckpointIO

from typing import Any, Callable, Dict, Optional
from pytorch_lightning.utilities.types import _PATH
from ldm.pruner import prune_checkpoint
 
class PruningCheckpointIO(TorchCheckpointIO):
    def __init__(self, float32=False, token_classes=None):
        self.float32 = float32
        self.token_classes = token_classes

    def save_checkpoint(self, checkpoint, path, storage_options=None):
        pruned_checkpoint = prune_checkpoint(checkpoint, float32=self.float32, token_classes=self.token_classes)
        TorchCheckpointIO.save_checkpoint(self, pruned_checkpoint, path, storage_options)

