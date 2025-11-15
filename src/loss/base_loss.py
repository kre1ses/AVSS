from abc import ABC, abstractmethod
import torch
from torch import Tensor

class BaseLoss(ABC):
    """
    Base class for all losses
    """

    def __init__(self, name=None,  *args, **kwargs):
        """
        Args:
            name (str | None): loss name
        """
        self.name = name if name is not None else type(self).__name__

    @abstractmethod
    def calc_loss(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Input: pred, target: [B, T]
        Output: loss values 
        """
        raise NotImplementedError


    def __call__(
        self,
        s1_pred: Tensor,
        s2_pred: Tensor,
        s1_audio: Tensor,
        s2_audio: Tensor,
        **batch
    ) -> Tensor:
        """
        Permutance invariant training implementation
        s1_pred, s2_pred:  predicted waveforms [B, T]
        s1_audio, s2_audio: target waveforms [B, T]
        """
        m11 = self.calc_loss(s1_pred, s1_audio) 
        m22 = self.calc_loss(s2_pred, s2_audio)
        fn_1 = 0.5 * (m11 + m22)

        m12 = self.calc_loss(s1_pred, s2_audio)
        m21 = self.calc_loss(s2_pred, s1_audio)
        fn_2 = 0.5 * (m12 + m21)
        final = torch.min(fn_1, fn_2)
        return {'loss': torch.mean(final)}


