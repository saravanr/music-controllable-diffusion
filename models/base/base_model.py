import torch
from pytorch_lightning import LightningModule

class BaseModel(LightningModule):
    """
    Base model class that will be inherited by all model types
    """

    def __init__(self, lr=1e-4):
        self._lr = lr
