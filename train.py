import argparse
import glob
import torch
import torch.nn.functional as F
import os
import numpy as np

from typing import Optional
from tqdm.auto import tqdm
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler
from diffusers.training_utils import EMAModel
from accelerate import Accelerator
from diffusers.hub_utils import init_git_repo, push_to_hub

from config import config
from dataset import WikiArtDataset
from model import LatentUNet
from pipeline import LatentDDPMPipeline
from evaluate import evaluate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_loop(config: object, 
               model: object, 
               noise_scheduler: object, 
               optimizer: object, 
               train_dataloader: object, 
               lr_scheduler: object
               ) -> None:
    r"""Training loop.
        Args:
            config: static class with configurations.
            model: LatentUNet model.
            noise_scheduler: diffusion noise scheduler.
            optimizer: optimizer.
            train_dataloader: train data loader.
            lr_scheduler: learning rate scheduler.
        Returns:
            None.
     """
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(mixed_precision=config.mixed_precision,
                              gradient_accumulation_steps=config.gradient_accumulation_steps, 
                              log_with="tensorboard",
                              logging_dir=os.path.join(config.output_dir, "logs"))

    if accelerator.is_main_process:
        if config.push_to_hub:
            repo = init_git_repo(config, at_init=True)
        accelerator.init_trackers("train_example")
    
    # prepare everything
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, 
                                                                           optimizer, 
                                                                           train_dataloader, 
                                                                           lr_scheduler)
    
    ema_model = EMAModel(accelerator.unwrap_model(model),
                         inv_gamma=config.ema_inv_gamma,
                         power=config.ema_power,
                         max_value=config.ema_max_decay)
    
    global_step = model.get_step()
    current_epoch = int(np.ceil(global_step / len(train_dataloader)))

    # Now you train the model
    for epoch in range(current_epoch + 1, config.num_epochs):

        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images, artist_label, genre_label, style_label = batch
            clean_images = clean_images.to(device)
            artist_label, genre_label, style_label = artist_label.to(device), genre_label.to(device), style_label.to(device)

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timestep=timesteps, 
                                   artist_class_labels=artist_label, 
                                   genre_class_labels=genre_label, 
                                   style_class_labels=style_label, 
                                   return_dict=False)[0]

                loss = F.huber_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                
                if config.use_ema:
                    ema_model.step(model)
                    
                optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            
            if config.use_ema:
                logs["ema_decay"] = ema_model.decay
                
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process and ((global_step + 1) % config.save_model_steps == 0 or epoch == config.num_epochs - 1):
                pipeline = LatentDDPMPipeline(unet=accelerator.unwrap_model(ema_model.averaged_model if config.use_ema else model), scheduler=noise_scheduler)

                if config.save_image_steps != -1 and ((global_step + 1) % config.save_image_steps == 0 or epoch == config.num_epochs - 1):
                    evaluate(config, step, pipeline)

                if (global_step + 1) % config.save_model_steps == 0 or epoch == config.num_epochs - 1:
                    if config.push_to_hub:
                        push_to_hub(config, pipeline, repo, commit_message=f"Epoch {epoch}", blocking=True)
                    else:
                        pipeline.save_pretrained(config.output_dir) 


def train(force_restart: Optional[bool]=False) -> None:
    r"""Main train function.
        Args:
            force_restart: starts training from scratch if True.
        Returns:
            None.
     """
    samples_paths = glob.glob(config.dataset_path + '/*.pickle')

    dataset = WikiArtDataset(samples_paths)

    train_dataloader = torch.utils.data.DataLoader(dataset, 
                                                   batch_size=config.train_batch_size, 
                                                   shuffle=True,
                                                   num_workers=config.num_workers,
                                                   pin_memory=True,
                                                   drop_last=False)

    model = LatentUNet(sample_size=config.sample_size,
                       in_channels=config.in_channels,
                       out_channels=config.out_channels,
                       layers_per_block=config.layers_per_block,
                       block_out_channels=config.block_out_channels, 
                       down_block_types=config.down_block_types, 
                       up_block_types=config.up_block_types,
                       num_artist_class_embeds = config.num_artist_classes,
                       num_genre_class_embeds = config.num_genre_classes,
                       num_style_class_embeds = config.num_style_classes,
                       class_embed_dropout1=config.class_embed_dropout,
                       class_embed_dropout2=config.class_embed_dropout).to(device)

    if not force_restart and os.path.exists(config.pretrained_path):
        model = model.from_pretrained(pretrained_model_name_or_path = config.pretrained_path).to(device)
        print("Model weights loaded from step %d" % model.step)

    params = []

    for name, values in model.named_parameters():

        if 'bias' not in name and 'conv' in name:
            params += [{'params': [values], 'lr': config.learning_rate, 'weight_decay': config.weight_decay}]
        elif 'bias' not in name and 'prenet' in name:
            params += [{'params': [values], 'lr': config.learning_rate, 'weight_decay': config.weight_decay}]
        else:
            params += [{'params': [values], 'lr': config.learning_rate, 'weight_decay': 0.0}]

    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, 
                                  betas=(config.adam_beta1, config.adam_beta2), 
                                  eps=config.adam_epsilon)

    lr_scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                   num_warmup_steps=config.lr_warmup_steps,
                                                   num_training_steps=(len(train_dataloader) * config.num_epochs))

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
    
    
def main():
    # Parse and read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--force_restart', default=False, type=bool, 
                        help='Start training and rewrite existing checkpoins.')
    args = parser.parse_args()
    
    train(args.force_restart)
    

if __name__ == "__main__":
    main()
