from torch import nn
import torch

class ThresholdClassifier(nn.Module):
    def __init__(self, audio_model, text_model, D) -> None:
        super().__init__()
        self.D = D
        self.audio_model = audio_model
        self.text_model = text_model