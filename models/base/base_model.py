from typing import Any

import torch
import os
from pytorch_lightning import LightningModule
from data.midi_data_module import MidiDataModule


class BaseModel(LightningModule):
    """
    Base model class that will be inherited by all model types
    """

    def __init__(self, lr=1e-1, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        data_dir = os.path.expanduser("~/midi/")
        self._lr = lr
        self.dms = MidiDataModule(data_dir)
