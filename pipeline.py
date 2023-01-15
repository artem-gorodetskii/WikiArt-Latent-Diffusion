import torch

from typing import List, Optional, Tuple, Union
from diffusers.configuration_utils import FrozenDict
from diffusers.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.utils import deprecate
from tqdm import tqdm


class LatentDDPMPipeline(DiffusionPipeline):
    r"""Pipeline for latent diffusion.
        Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/ddpm/pipeline_ddpm.py
        Copyright 2022 The HuggingFace Team. All rights reserved.
     """
    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = None,
        artist_class_labels: Optional[List[int]] = None,
        genre_class_labels: Optional[List[int]] = None,
        style_class_labels: Optional[List[int]] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:

        message = ("Please make sure to instantiate your scheduler with `prediction_type` instead. E.g. `scheduler ="
                   " DDPMScheduler.from_pretrained(<model_id>, prediction_type='epsilon')`.")

        predict_epsilon = deprecate("predict_epsilon", "0.13.0", message, take_from=kwargs)

        if predict_epsilon is not None:
            new_config = dict(self.scheduler.config)
            new_config["prediction_type"] = "epsilon" if predict_epsilon else "sample"
            self.scheduler._internal_dict = FrozenDict(new_config)

        if generator is not None and generator.device.type != self.device.type and self.device.type != "mps":
            message = (f"The `generator` device is `{generator.device}` and does not match the pipeline "
                       f"device `{self.device}`, so the `generator` will be ignored. "
                       f'Please use `torch.Generator(device="{self.device}")` instead.')

            deprecate("generator.device == 'cpu'",
                      "0.13.0",
                      message)
                      
            generator = None

        if artist_class_labels is None or genre_class_labels is None or style_class_labels is None:
            raise ValueError("class_labels should be provided")

        artist_labels = torch.LongTensor(artist_class_labels).to(self.device)
        genre_labels = torch.LongTensor(genre_class_labels).to(self.device)
        style_labels = torch.LongTensor(style_class_labels).to(self.device)

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.sample_size, int):
            sample_shape = (batch_size, self.unet.in_channels, self.unet.sample_size, self.unet.sample_size)
        else:
            sample_shape = (batch_size, self.unet.in_channels, *self.unet.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            sample = torch.randn(sample_shape, generator=generator)
            sample = sample.to(self.device)
        else:
            sample = torch.randn(sample_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(self.scheduler.timesteps):
            # 1. predict noise model_output
            model_output = self.unet(sample, timestep=t, 
                                     artist_class_labels=artist_labels, 
                                     genre_class_labels=genre_labels, 
                                     style_class_labels=style_labels).sample

            # 2. compute previous image: x_t -> x_t-1
            sample = self.scheduler.step(model_output, t, sample, generator=generator).prev_sample

        if not return_dict:
            return (sample,)

        return ImagePipelineOutput(images=sample)
