import numpy as np
import torch
import pickle
import os

from typing import Dict
from pathlib import Path
from skimage import transform
from tqdm.auto import tqdm
from datasets import load_dataset
from diffusers import StableDiffusionPipeline

from config import config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(sample: Dict) -> np.ndarray:
    r"""Preprocesses the image. Applies cropping and resizing.
        Args:
            sample: dataset sample.
        Returns:
            Image as np array.
     """
    img = np.asarray(sample['image'])
    h, w, c = img.shape 
    
    if h >= w:
        img = transform.resize(img, (int(config.image_size * h / w), config.image_size, c), anti_aliasing=True)
        delta = (int(config.image_size * h / w) - config.image_size) // 2
        img = img[:config.image_size, : , :]
    else:
        img = transform.resize(img, (config.image_size, int(config.image_size * w / h), c), anti_aliasing=True)
        delta = (int(config.image_size * w / h) - config.image_size) // 2
        img = img[:, delta : config.image_size + delta , :]
        
    # check 
    new_h, new_w, _ = img.shape
    
    if new_h != config.image_size or new_w != config.image_size:
        img = transform.resize(img, (config.image_size, config.image_size, c), anti_aliasing=True)
    
    return img


def generate() -> None:
    r"""Creates dataset. Computes the latent representation of the image, extracts labels
        and saves samples as a pickle file. 
        Returns:
            None.
     """
    dataset_dir = Path(config.dataset_dir)
    dataset_dir.mkdir(exist_ok=True)

    dataset = load_dataset(config.dataset_name, split="train")

    pipe = StableDiffusionPipeline.from_pretrained(config.stable_diffusion_id).to(device)
    
    for index, sample in enumerate(tqdm(dataset)):
        prosessed_sample = preprocess(sample)

        vae_inputs = torch.FloatTensor(prosessed_sample).unsqueeze(0).permute(0, 3, 1, 2).to(device) * 2 - 1 

        # Encode to latent space
        with torch.no_grad():
            latents = config.vae_scale * pipe.vae.encode(vae_inputs).latent_dist.mean

        features = {'latent': latents[0].detach().cpu().numpy(),
                    'artist': sample['artist'], 
                    'genre': sample['genre'], 
                    'style': sample['style']}

        sample_name = '{}.pickle'.format(index)

        with open(os.path.join(config.dataset_dir, sample_name), 'wb') as handle:
            pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
if __name__ == "__main__":
    generate()
    