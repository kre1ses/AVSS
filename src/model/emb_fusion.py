import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import Tensor


class LinearFusion(nn.Module):
    """
    Performs linear fusion of audio and video embeddings

    Args:
        audio_emb_len (int): audio embedding length
        video_emb_len (int): embedding len of 1 video
        audio_len (int): audio length for video embedding to be interpolated
        interpolation_mode (str): defines how video embs will be interpolated to audio_len

    Input:
        [batch, audio_emb_len, audio_len] - audio embeddings
        [batch, video_emb_len, video_len] - s1_video_embs, s2_video_embs

    Output: [batch, audio_emb_len, audio_len] - fused embeddings
    """

    def __init__(
        self,
        audio_emb_len: int,
        video_emb_len: int,
        audio_len: int,
        interpolation_mode: str,
    ):
        super().__init__()
        assert audio_emb_len % 2 == 0, "audio_emb_len must be divisible by 2"
        assert interpolation_mode in ["linear", "nearest", "nearest-exact"]
        self.audio_len = audio_len
        self.interpolation_mode = interpolation_mode

        self.video_projection_s1 = nn.Linear(video_emb_len, audio_emb_len // 2)
        self.video_projection_s2 = nn.Linear(video_emb_len, audio_emb_len // 2)
        self.activation = nn.GELU()
        self.final_projection = nn.Linear(audio_emb_len * 2, audio_emb_len)

    def forward(
        self, audio_embs: Tensor, s1_video_embs: Tensor, s2_video_embs: Tensor
    ) -> Tensor:
        proj_s1 = self.video_projection_s1(
            s1_video_embs.permute(0, 2, 1)
        )  # [batch, video_len, audio_emb_len // 2]
        proj_s2 = self.video_projection_s2(
            s2_video_embs.permute(0, 2, 1)
        )  # [batch, video_len, audio_emb_len // 2]

        video_emb = torch.concat(
            [proj_s1, proj_s2], dim=-1
        )  # [batch, video_len, audio_emb_len]
        video_emb = self.activation(video_emb)
        interpolated_video_embs = f.interpolate(
            video_emb.permute(0, 2, 1),
            size=self.audio_len,
            mode=self.interpolation_mode,
        ).permute(
            0, 2, 1
        )  # [batch, audio_len, audio_emb_len]

        audio_video_emb = torch.concat(
            [audio_embs.permute(0, 2, 1), interpolated_video_embs], dim=-1
        )  # [batch, audio_len, audio_emb_len * 2]
        audio_video_emb = self.activation(audio_video_emb)

        final_emb = self.final_projection(audio_video_emb).permute(0, 2, 1)

        return final_emb


class GatedFusion(nn.Module):
    """
    Performs Gated embedding fusion of audio and video embeddings

    Args:
        audio_emb_len (int): audio embedding length
        video_emb_len (int): embedding len of 1 video
        audio_len (int): audio length for video embedding to be interpolated
        interpolation_mode (str): defines how video embs will be interpolated to audio_len

    Input:
        [batch, audio_emb_len, audio_len] - audio embeddings
        [batch, video_emb_len, video_len] - s1_video_embs, s2_video_embs

    Output: [batch, audio_emb_len, audio_len] - fused embeddings
    """

    def __init__(
        self,
        audio_emb_len: int,
        video_emb_len: int,
        audio_len: int,
        interpolation_mode: str,
    ):
        super().__init__()
        assert audio_emb_len % 2 == 0, "audio_emb_len must be divisible by 2"
        assert interpolation_mode in ["linear", "nearest", "nearest-exact"]
        self.audio_len = audio_len
        self.interpolation_mode = interpolation_mode

        self.video_projection_s1 = nn.Linear(video_emb_len, audio_emb_len // 2)
        self.video_projection_s2 = nn.Linear(video_emb_len, audio_emb_len // 2)
        self.gating_projection = nn.Linear(audio_emb_len * 2, audio_emb_len)

    def forward(
        self, audio_embs: Tensor, s1_video_embs: Tensor, s2_video_embs: Tensor
    ) -> Tensor:
        proj_s1 = self.video_projection_s1(
            s1_video_embs.permute(0, 2, 1)
        )  # [batch, video_len, audio_emb_len // 2]
        proj_s2 = self.video_projection_s2(
            s2_video_embs.permute(0, 2, 1)
        )  # [batch, video_len, audio_emb_len // 2]

        video_emb = torch.concat(
            [proj_s1, proj_s2], dim=-1
        )  # [batch, video_len, audio_emb_len]
        interpolated_video_embs = f.interpolate(
            video_emb.permute(0, 2, 1),
            size=self.audio_len,
            mode=self.interpolation_mode,
        ).permute(
            0, 2, 1
        )  # [batch, audio_len, audio_emb_len]
        audio_video_emb = torch.concat(
            [audio_embs.permute(0, 2, 1), interpolated_video_embs], dim=-1
        )  # [batch, audio_len, audio_emb_len * 2]
        gating = (
            self.gating_projection(audio_video_emb).sigmoid().permute(0, 2, 1)
        )  # [batch, audio_emb_len, audio_len]

        final_emb = gating * audio_embs + (
            1 - gating
        ) * interpolated_video_embs.permute(0, 2, 1)

        return final_emb


class AttentionFusion(nn.Module):
    """
    Performs atttention based fusion of audio and video embeddings

    Args:
        audio_emb_len (int): audio embedding length
        video_emb_len (int): embedding len of 1 video
        audio_len (int): audio length for video embedding to be interpolated
        interpolation_mode (str): defines how video embs will be interpolated to audio_len
        num_heads (int): number of heads in attention module
        emb_dropout (float): dropout in attention module
        bidir (bool): defines whether only audio embeddings are enriched through attention or both

    Input:
        [batch, audio_emb_len, audio_len] - audio embeddings
        [batch, video_emb_len, video_len] - s1_video_embs, s2_video_embs

    Output: [batch, audio_emb_len, audio_len] - fused embeddings

    """

    def __init__(
        self,
        audio_emb_len: int,
        video_emb_len: int,
        audio_len: int,
        interpolation_mode: str,
        d_model: int,
        num_heads: int,
        emb_dropout: float,
        bidir: bool,
    ):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be divisible by 2"
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.audio_len = audio_len
        self.interpolation_mode = interpolation_mode
        self.bidir = bidir

        self.video_projection_s1 = nn.Linear(video_emb_len, d_model // 2)
        self.video_projection_s2 = nn.Linear(video_emb_len, d_model // 2)
        self.audio_projection = nn.Linear(audio_emb_len, d_model)
        self.out_proj = nn.Linear(d_model * 2, audio_emb_len)

        if bidir:
            self.audio_video_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, dropout=emb_dropout
            )
            self.video_audio_attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, dropout=emb_dropout
            )
        else:
            self.attn = nn.MultiheadAttention(
                embed_dim=d_model, num_heads=num_heads, dropout=emb_dropout
            )

    def forward(
        self, audio_embs: Tensor, s1_video_embs: Tensor, s2_video_embs: Tensor
    ) -> Tensor:
        proj_audio = self.audio_projection(
            audio_embs.permute(0, 2, 1)
        )  # [batch, audio_len, d_model]
        proj_s1 = self.video_projection_s1(
            s1_video_embs.permute(0, 2, 1)
        )  # [batch, video_len, d_model // 2]
        proj_s2 = self.video_projection_s2(
            s2_video_embs.permute(0, 2, 1)
        )  # [batch, video_len, d_model // 2]

        video_emb = torch.concat(
            [proj_s1, proj_s2], dim=-1
        )  # [batch, video_len, d_model]
        interpolated_video_embs = f.interpolate(
            video_emb.permute(0, 2, 1),
            size=self.audio_len,
            mode=self.interpolation_mode,
        )  # [batch, d_model, audio_len]

        audio_attn = proj_audio.permute(1, 0, 2)  # [audio_len, batch, d_model]
        video_attn = interpolated_video_embs.permute(
            2, 0, 1
        )  # [audio_len, batch, d_model]

        if self.bidir:
            video_attn_embs, _ = self.audio_video_attn(
                query=audio_attn, key=video_attn, value=video_attn
            )  # [audio_len, batch, d_model]
            audio_attn_embs, _ = self.video_audio_attn(
                query=video_attn, key=audio_attn, value=audio_attn
            )  # [audio_len, batch, d_model]

            long_embed = torch.concat(
                [audio_attn_embs.permute(1, 0, 2), video_attn_embs.permute(1, 0, 2)],
                dim=-1,
            )  # [batch, audio_len, d_model * 2]
            final_embed = (
                self.out_proj(long_embed).permute(0, 2, 1) + audio_embs
            )  # [batch, audio_emb_len, audio_len]

            return final_embed

        video_attn_embs, _ = self.attn(
            query=audio_attn, key=video_attn, value=video_attn
        )  # [audio_len, batch, d_model]
        long_embed = torch.concat(
            [proj_audio, video_attn_embs.permute(1, 0, 2)], dim=-1
        )  # [batch, audio_len, d_model * 2]
        final_embed = (
            self.out_proj(long_embed).permute(0, 2, 1) + audio_embs
        )  # [batch, audio_emb_len, audio_len]

        return final_embed
