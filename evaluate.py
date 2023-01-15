import torch
import os

from torch import tensor as Tensor
from typing import Optional, List
from PIL import Image
from diffusers import StableDiffusionPipeline

from config import config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load StableDiffusion pipeline
# It may take 5 minutes for the first time
STABLE_DIFFUSION_PIPE = StableDiffusionPipeline.from_pretrained(config.stable_diffusion_id).to(device)


def latents_to_pil(latents: Tensor) -> List:
    r"""Converts a batch of latents to list of images.
        Args:
            latents: torch tensor.
        Returns:
            list of PIL images.
     """
    with torch.no_grad():
        image = STABLE_DIFFUSION_PIPE.vae.decode(latents / config.vae_scale).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def make_grid(images: List, rows: int, cols: int):
    r"""Combines a list of PIL images into one image.
        Args:
            images: list of PIL images.
            rows: number of rows.
            cols: number of columns.
        Returns:
            PIL image.
     """
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))

    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))

    return grid


def evaluate(config: object, step: int, pipeline: object, save: Optional[bool]=True, 
             return_pil: Optional[bool]=False, seed: Optional[int]=config.seed):
    r"""Evaluates pre-trained pipeline.
        Args:
            config: static configuration class.
            step: number of training steps.
            pipeline: pipeline, instance of LatentDDPMPipeline class.
            save: saves image as jpg if True.
            return_pil: returns PIL image if True.
            seed: random seed.
        Returns:
            None or PIL image.
     """
     # batch of labels from configuration file
    eval_artist_class_labels = [config.artists.index(name) for name in config.evaluation_artists]
    eval_genre_class_labels = [config.genre.index(name) for name in config.evaluation_genres]
    eval_style_class_labels = [config.style.index(name) for name in config.evaluation_styles]

    # generate latent representations
    samples = pipeline(batch_size = config.eval_batch_size, 
                       generator=torch.manual_seed(seed),
                       artist_class_labels=eval_artist_class_labels,
                       genre_class_labels=eval_genre_class_labels,
                       style_class_labels=eval_style_class_labels).images

    # convert latent representation to images
    images = latents_to_pil(samples)

    # make a grid out of the images
    image_grid = make_grid(images, rows=2, cols=4)

    # save the images
    if save:
      test_dir = os.path.join(config.output_dir, "samples")
      os.makedirs(test_dir, exist_ok=True)
      image_grid.save(f"{test_dir}/{step / 1000}k.png")

    # return image
    if return_pil:
      return image_grid
