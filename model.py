import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Union
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from diffusers.models.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block
from diffusers.models.unet_2d import UNet2DOutput


class PreNet(nn.Module):
    r"""PreNet class. 
        Preprocesses label embedding.
     """
    def __init__(
        self, 
        in_dims: Optional[int] = 512, 
        dropout1: Optional[float] = 0.5, 
        dropout2: Optional[float] = 0.5
        ):

        super().__init__()
        self.fc1 = nn.Linear(in_dims, 4*in_dims)
        self.fc2 = nn.Linear(4*in_dims, in_dims)
        self.p1 = dropout1
        self.p2 = dropout2

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, self.p1, training=True)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, self.p2, training=True)

        return x

    
class LatentUNet(ModelMixin, ConfigMixin):
    r"""LatentUNet is a 2D UNet model that takes in a noisy sample of latent representation, 
        a timestep and class labels and returns sample shaped output.
        Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d.py.
        Copyright 2022 The HuggingFace Team. All rights reserved.
     """
    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str] = ("AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types: Tuple[str] = ("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D"),
        block_out_channels: Tuple[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        act_fn: str = "silu",
        attention_head_dim: int = 8,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        class_embed_type: Optional[str] = None,
        num_artist_class_embeds: Optional[int] = None,
        num_genre_class_embeds: Optional[int] = None,
        num_style_class_embeds: Optional[int] = None,
        class_embed_dropout1: Optional[float] = 0.1,
        class_embed_dropout2: Optional[float] = 0.1,
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))

        # time
        if time_embedding_type == "fourier":
            self.time_proj = GaussianFourierProjection(embedding_size=block_out_channels[0], scale=16)
            timestep_input_dim = 2 * block_out_channels[0]
        elif time_embedding_type == "positional":
            self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
            timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embeddings
        if class_embed_type is None and num_artist_class_embeds is not None:
            self.artist_class_embedding = nn.Embedding(num_artist_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.artist_class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.artist_class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.artist_class_embedding = None

        if class_embed_type is None and num_genre_class_embeds is not None:
            self.genre_class_embedding = nn.Embedding(num_genre_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.genre_class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.genre_class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.genre_class_embedding = None

        if class_embed_type is None and num_style_class_embeds is not None:
            self.style_class_embedding = nn.Embedding(num_style_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.style_class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.style_class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.style_class_embedding = None
            
        self.artist_prenet = PreNet(time_embed_dim, class_embed_dropout1, class_embed_dropout2)
        self.genre_prenet = PreNet(time_embed_dim, class_embed_dropout1, class_embed_dropout2)
        self.style_prenet = PreNet(time_embed_dim, class_embed_dropout1, class_embed_dropout2)

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(down_block_type,
                                        num_layers=layers_per_block,
                                        in_channels=input_channel,
                                        out_channels=output_channel,
                                        temb_channels=time_embed_dim,
                                        add_downsample=not is_final_block,
                                        resnet_eps=norm_eps,
                                        resnet_act_fn=act_fn,
                                        resnet_groups=norm_num_groups,
                                        attn_num_head_channels=attention_head_dim,
                                        downsample_padding=downsample_padding,
                                        resnet_time_scale_shift=resnet_time_scale_shift)

            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlock2D(in_channels=block_out_channels[-1],
                                        temb_channels=time_embed_dim,
                                        resnet_eps=norm_eps,
                                        resnet_act_fn=act_fn,
                                        output_scale_factor=mid_block_scale_factor,
                                        resnet_time_scale_shift=resnet_time_scale_shift,
                                        attn_num_head_channels=attention_head_dim,
                                        resnet_groups=norm_num_groups,
                                        add_attention=add_attention)

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]

        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            up_block = get_up_block(up_block_type,
                                    num_layers=layers_per_block + 1,
                                    in_channels=input_channel,
                                    out_channels=output_channel,
                                    prev_output_channel=prev_output_channel,
                                    temb_channels=time_embed_dim,
                                    add_upsample=not is_final_block,
                                    resnet_eps=norm_eps,
                                    resnet_act_fn=act_fn,
                                    resnet_groups=norm_num_groups,
                                    attn_num_head_channels=attention_head_dim,
                                    resnet_time_scale_shift=resnet_time_scale_shift)
                                    
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        num_groups_out = norm_num_groups if norm_num_groups is not None else min(block_out_channels[0] // 4, 32)
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

        # Initialize 'step' variable - number of model forward pass
        self.register_buffer("step", torch.zeros(1, dtype=torch.long))

        self.num_params()

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        artist_class_labels: Optional[torch.Tensor] = None,
        genre_class_labels: Optional[torch.Tensor] = None,
        style_class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DOutput, Tuple]:

        # Count the number of forward pass
        self.step += 1

        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.artist_class_embedding is not None and self.genre_class_embedding is not None and self.style_class_embedding is not None:
            if artist_class_labels is None or genre_class_labels is None or style_class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                artist_class_labels = self.time_proj(artist_class_labels)
                genre_class_labels = self.time_proj(genre_class_labels)
                style_class_embedding = self.time_proj(style_class_embedding)

            artist_class_emb = self.artist_class_embedding(artist_class_labels).to(dtype=self.dtype)
            genre_class_emb = self.genre_class_embedding(genre_class_labels).to(dtype=self.dtype)
            style_class_emb = self.style_class_embedding(style_class_labels).to(dtype=self.dtype)

            artist_class_emb = artist_class_emb.squeeze(1)
            genre_class_emb = genre_class_emb.squeeze(1)
            style_class_emb = style_class_emb.squeeze(1)

            emb += self.artist_prenet(artist_class_emb)
            emb += self.genre_prenet(genre_class_emb)
            emb += self.style_prenet(style_class_emb)

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(hidden_states=sample, temb=emb, skip_sample=skip_sample)
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, emb)

        # 5. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        if not return_dict:
            return (sample,)

        return UNet2DOutput(sample=sample)

    def num_params(self, print_out: bool = True):
        """Counts number of trained parameters.
        Args:
            print_out: if True, the number of parameters in printed.
        """
        parameters = filter(lambda p: p.requires_grad, self.parameters())

        parameters = sum([np.prod(p.size()) for p in parameters]) / 1000000
        if print_out:
            print('Trainable Parameters: %.3fM' % parameters)

    def get_step(self):
        return self.step.data.item()
      
    def set_step(self, value):
        self.step = self.step.data.new_tensor([value])
    