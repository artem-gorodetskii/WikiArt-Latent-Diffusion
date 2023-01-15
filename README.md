# WikiArt-Latent-Diffusion
Conditional denoising diffusion probabilistic model trained in latent space to generate paintings by famous artists. See the animation of the latent diffusion process in the figure below.

<p align="center">
  <img alt="img-name" src="assets/inference_gif.gif" width="500">
  <br>
    <em>Fig. 1. The animation of the latent diffusion process.</em>
</p>

### Repository structure:
- **[config.py](https://github.com/artem-gorodetskii/WikiArt-Latent-Diffusion/blob/master/config.py)** is a file with model hyperparameters.
- **[dataset.py](https://github.com/artem-gorodetskii/WikiArt-Latent-Diffusion/blob/master/dataset.py)** contains dataset class.
- **[generate_features.py](https://github.com/artem-gorodetskii/WikiArt-Latent-Diffusion/blob/master/generate_features.py)** contains functions to prepare dataset.
- **[models.py](https://github.com/artem-gorodetskii/WikiArt-Latent-Diffusion/blob/master/model.py)** contains implementations of the latent UNet model.
- **[pipeline.py](https://github.com/artem-gorodetskii/WikiArt-Latent-Diffusion/blob/master/pipeline.py)** is a latent diffusion pipeline.
- **[train.py](https://github.com/artem-gorodetskii/WikiArt-Latent-Diffusion/blob/master/train.py)** performs training of the LatentUNet model using a single GPU instance.
- **[evaluate.py](https://github.com/artem-gorodetskii/WikiArt-Latent-Diffusion/blob/master/evaluate.py)** performs evaluation of trained pipeline.
- the notebook **[inference_example](https://github.com/artem-gorodetskii/WikiArt-Latent-Diffusion/blob/master/inference_example.ipynb)** includes inference examples of the developed pieline.

