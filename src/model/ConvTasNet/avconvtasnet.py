import torch.nn as nn
from torch import Tensor

from src.model.ConvTasNet.decoder import ConvTasNetDecoder
from src.model.ConvTasNet.encoder import ConvTasNetEncoder
from src.model.ConvTasNet.separator import ConvTasNetSeparator
from src.model.emb_fusion import LinearFusion
from src.model.emb_fusion import GatedFusion
from src.model.emb_fusion import AttentionFusion


class AVConvTasNet(nn.Module):
    """
    ConvTasNet model with Video Support (https://arxiv.org/pdf/1809.07454)

    Args:
        N (int): number of filters in autoencoder
        L (int): length of the filters (in samples)
        B (int): number of channels in bottleneck and residual paths
        Sc (int): number of channels in skip-connection paths
        H (int): number of channels in conv blocks
        P (int): kernel size in conv block
        X (int): number of conv blocks in repeat
        R (int): number of repeats
        C (int): number of speakers
        fusion_method (str): defines how audio and video embeddings will be fused
        audio_len (int): audio length
        interpolation_mode (str): defines the way video embeddings are interpolated to audio embeddings length
        d_model (int): hidden dimension for attention fusion
        num_heads (int): amount of heads in attention module in attention fusion
        emb_dropout (float): dropout prob in attention module in attention fusion
        bidir (bool): defines whether only audio embeddings are enriched through attention or both 

    Input: [batch, T]
    Output: dict, where s{i} -> [:, i, :] (i-th speaker audio)
    """

    def __init__(
        self,
        N: int = 512,
        L: int = 16,
        B: int = 128,
        Sc: int = 128,
        H: int = 512,
        P: int = 3,
        X: int = 8,
        R: int = 3,
        C: int = 2,
        fusion_method: str = 'linear',
        audio_len: int = 3999, # не уверен
        interpolation_mode: str = 'linear',
        d_model: int = 256,
        num_heads: int = 4,
        emb_dropout: float = 0.1,
        bidir: bool = False,
    ):
        super().__init__()

        assert fusion_method in ['linear', 'gated', 'attention'], 'fusion method must be linear, gated or attention based'

        if fusion_method == 'linear':
            self.fusion = LinearFusion(audio_emb_len=N, 
                                        video_emb_len=512, 
                                        audio_len=audio_len, 
                                        interpolation_mode=interpolation_mode
                                    )
            
        elif fusion_method == 'gated':
            self.fusion = GatedFusion(audio_emb_len=N, 
                                        video_emb_len=512, 
                                        audio_len=audio_len, 
                                        interpolation_mode=interpolation_mode
                                    )
        
        else:
            self.fusion = AttentionFusion(audio_emb_len=N, 
                                        video_emb_len=512, 
                                        audio_len=audio_len, 
                                        interpolation_mode=interpolation_mode,
                                        d_model=d_model,
                                        num_heads=num_heads,
                                        emb_dropout=emb_dropout,
                                        bidir=bidir
                                    )

        self.encoder = ConvTasNetEncoder(N, L)
        self.separator = ConvTasNetSeparator(N, B, Sc, H, P, X, R, C)
        self.decoder = ConvTasNetDecoder(N, L)

    def forward(self, mix_audio: Tensor, s1_embs: Tensor, s2_embs: Tensor, **batch) -> dict:
        
        mix_audio = mix_audio.unsqueeze(1)  # [batch, 1, T]
        mix_enc = self.encoder(mix_audio)  # [batch, N, T_new]
        fused_embs = self.fusion(mix_enc, s1_embs, s2_embs)
        masks = self.separator(fused_embs)  # [batch, C, N, T_new]
        masked_audios = fused_embs.unsqueeze(1) * masks  # [batch, C, N, T_new]
        separated_audios = self.decoder(masked_audios)  # [batch, C, T]

        return {
            "s1_pred": separated_audios[:, 0, :],
            "s2_pred": separated_audios[:, 1, :],
        }
