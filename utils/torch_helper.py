from typing import Any
import numpy as np
import torch
class ToTensor:
    def __call__(self, x: np.array) -> torch.Tensor:
        return torch.from_numpy(x)

# Hyperparameters
class Param:
    INPUT_SIZE: int= 784
    HIDDEN_SIZE:int = 100
    OUTPUT_SIZE:int = 10

class OPS:
    LR:float = 0.001
    BATCH_SIZE:int = 100
    NUM_EPOCHS:int = 2
