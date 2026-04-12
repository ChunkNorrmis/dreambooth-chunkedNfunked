from pytorch_lightning.plugins.io.torch_plugin import TorchCheckpointIO
import safetensors.torch.save_model

from typing import Any, Callable, Dict, Optional
from pytorch_lightning.utilities.types import _PATH
from ldm.pruner import prune_checkpoint
 
class PruningCheckpointIO(TorchCheckpointIO):
    def __init__(self, precision='float16', sftsr=False):
        self.precision = precision
    def save_checkpoint(
            self, 
            checkpoint: Dict[str, Any], 
            path: _PATH, 
            storage_options: Optional[Any] = None
        ) -> None:
        pruned_checkpoint = prune_checkpoint(checkpoint, precision=self.precision)
        if not sftsr:
            TorchCheckpointIO.save_checkpoint(self, pruned_checkpoint, path, storage_options)
        else:
            safetensors.torch.save_model(pruned_checkpoint, path)
