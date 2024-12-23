#!/usr/bin/env python3
import sys
#sys.path.append('/home/hlcv_team016/project/code')

import os
import numpy as np
import pandas as pd
import random
import warnings
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from torchvision import datasets, models, transforms

import PIL.Image as pil
from tqdm import tqdm
import matplotlib.pyplot as plt

# Project libraries
from model import *
from utils import * # metricsCalculator, psnr, save_history
import ssim
from dataloader import DIV2KDataset
from transforms import *
import lpips #lpips
# To supress the warnings at the output
warnings.filterwarnings("ignore")
# Empty the cache for CUDA memory
torch.cuda.empty_cache()
total_start_time = time.time()
#======================================================================================
# Device configuration
#======================================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)
#======================================================================================
# Directories to local files
#======================================================================================
# Used for dataloader
path_imgs = '/root/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_train_HR/DIV2K_train_HR'
test_hr = '/root/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_valid_HR/DIV2K_valid_HR'
local_path = ""


saved_weights_dir = 'saved_weights'
output_dir = 'train_output'

create_dir(saved_weights_dir)
create_dir(output_dir)
#======================================================================================
# Hyper-parameters
#======================================================================================
num_epochs = 100
lr = 0.0002
batch_size = 3
batch_size_val = 1
lambda_adv = 5e-3 #0.005
lambda_pixel = 1e-2 #0.01
lambda_lpips = 0.1
residual_blocks = 18
b1 = 0.9
b2 = 0.99
lr_size = 64
hr_size = 256

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])



# Training set
train_dataset = DIV2KDataset(path_imgs, transform=transform, crop_size=256)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,)
# Validation set
valid_dataset = DIV2KDataset(test_hr, transform=transform, crop_size=256)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

#======================================================================================
# Weight initialization
#======================================================================================
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity = 'relu')
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias.data, 0.0)
#======================================================================================
# Model
#======================================================================================
def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()

# Initialize generator and discriminator
generator = Generator(3, filters=128, num_res_blocks=residual_blocks).to(device)
discriminator = Discriminator(input_shape=(3, hr_size, hr_size)).to(device)
feature_extractor = FeatureExtractor().to(device)
# Set feature extractor to inference mode
feature_extractor.eval()
#model.apply(init_weights)
#======================================================================================
# Loss and optimizer
#======================================================================================
# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss().to(device)
criterion_content = torch.nn.L1Loss().to(device)
criterion_pixel = torch.nn.L1Loss().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_G,
                                                  factor=0.1,
                                                  mode='min',
                                                  min_lr=1e-10,
                                                  patience=7)
#======================================================================================
# Metrices dictionary
#======================================================================================
loss_history = {"epoch": [], "lr":[], "train loss": [], "val loss":[]}
psnr_metrices_history = {"epoch":[], "train psnr":[], "val psnr":[]}
ssim_metrices_history = {"epoch":[], "train ssim":[], "val ssim":[]}
#======================================================================================
# Validation loop definition
#======================================================================================
import wandb

# Инициализация W&B
wandb.init(
    project='super_resolution',
    name='training_run',
    config={
        "num_epochs": num_epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "lambda_adv": lambda_adv,
        "lambda_pixel": lambda_pixel,
        "lambda_lpips": lambda_lpips,
        "residual_blocks": residual_blocks
    }
)

# Логирование модели
wandb.watch(generator, log="all")
wandb.watch(discriminator, log="all")

def val(epoch_num):
  generator.eval()

  epoch_val_psnr = MetricsCalculator()
  epoch_val_ssim = MetricsCalculator()
  example_images = []
  with torch.no_grad():
      for iteration, samples in tqdm(enumerate(valid_loader)):
          imgs_lr, imgs_hr = samples

          imgs_lr = imgs_lr.to(device)
          imgs_hr = imgs_hr.to(device)

          # Generate a high resolution image from low resolution input
          gen_hr = generator(imgs_lr)

          epoch_val_psnr.update(psnr(gen_hr,imgs_hr))
          ssim_score = ssim.ssim(gen_hr,imgs_hr)
          epoch_val_ssim.update(ssim_score.item())

          # Visualise output after every 10th epoch
          if(epoch_num%1== 0 and iteration%10==0):

              imgs_lr = denormalize(imgs_lr).detach().cpu().permute(0,2,3,1).numpy()[0]
              imgs_hr = denormalize(imgs_hr).detach().cpu().permute(0,2,3,1).numpy()[0]
              gen_hr  = denormalize(gen_hr).detach().cpu().permute(0,2,3,1).numpy()[0]
              example_images.append(wandb.Image((imgs_lr[0]*255).astype(np.uint8), caption="Low Resolution"))
              example_images.append(wandb.Image((gen_hr[0]*255).astype(np.uint8), caption="Generated High Resolution"))
              example_images.append(wandb.Image((imgs_hr[0]*255).astype(np.uint8), caption="Original High Resolution"))
              f, ax1 = plt.subplots(3, figsize=(14,14))
              ax1[0].set_title('LR Image')
              ax1[1].set_title('Predicted HR Image')
              ax1[2].set_title('Original HR Image')

              ax1[0].imshow((imgs_lr*255).astype(np.uint8))
              ax1[1].imshow((gen_hr*255).astype(np.uint8))
              ax1[2].imshow((imgs_hr*255).astype(np.uint8))
              f.tight_layout()
              plt.savefig(os.path.join(output_dir, 'fig_{}_{}.png'.format(epoch_num, iteration)), dpi=500)
  wandb.log({
        "Validation Images": example_images
    }, step=epoch_num)
  generator.train()
  return epoch_val_psnr, epoch_val_ssim
#======================================================================================
# Train loop
#======================================================================================
best_loss = 1000
get_lpips = lpips.LPIPS(net='alex').to(device)
for epoch in range(num_epochs):
    
    epoch_start_time = time.time()

    epoch_train_loss = MetricsCalculator()
    epoch_train_loss_D = MetricsCalculator()

    generator.train()
    discriminator.train()

    for iteration, samples in tqdm(enumerate(train_loader),
                total=len(train_loader),
                leave=True,position=0,
                desc='Epoch: {}'.format(epoch+1)):
        imgs_lr, imgs_hr = samples

        optimizer_G.zero_grad()

        valid = torch.ones((imgs_hr.size(0), *discriminator.output_shape)).to(device)
        fake = torch.zeros((imgs_hr.size(0), *discriminator.output_shape)).to(device)

        # Move tensors to the configured device
        imgs_lr = imgs_lr.to(device)
        imgs_hr = imgs_hr.to(device)

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        psnr_score_train = psnr(gen_hr, imgs_hr)
        # Measure pixel-wise loss against ground truth
        loss_pixel = criterion_pixel(gen_hr, imgs_hr)

        # Extract validity predictions from discriminator
        pred_real = discriminator(imgs_hr).detach()
        pred_fake = discriminator(gen_hr).detach()

        # Adversarial loss (relativistic average GAN)
        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr).detach()
        real_features = feature_extractor(imgs_hr).detach()
        #print(gen_features.size())
        loss_content = criterion_content(gen_features, real_features)

        # # Total generator loss
        # loss_G = loss_content + lambda_adv * loss_GAN + lambda_pixel * loss_pixel
        #print(gen_hr.get_device(), imgs_hr.get_device())
        lpips_score = get_lpips(gen_hr, imgs_hr)
        #print(torch.mean(lpips_score), lpips_score)
        # Total generator loss
        loss_G = loss_content + lambda_adv * loss_GAN + lambda_pixel * loss_pixel + lambda_lpips * torch.mean(lpips_score).item()

        epoch_train_loss.update(loss_G.detach().item())

        # Backward and optimize
        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        gaussian_variance = fetch_gauss_variance(iteration)
        gaussian_noise = AddGaussianNoise(0, gaussian_variance)
        imgs_hr = gaussian_noise(imgs_hr)
        gen_hr = gaussian_noise(gen_hr)

        pred_real = discriminator(imgs_hr)
        pred_fake = discriminator(gen_hr.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2
        epoch_train_loss_D.update(loss_D.detach().item())
        loss_D.backward()
        optimizer_D.step()
    
    state = {'epoch': (epoch+1),'state_dict': generator.state_dict(),
                 'optimizer':  optimizer_G.state_dict(),'loss': epoch_train_loss.avg}
    #======================================================================================
    # Validation loop
    #======================================================================================
    epoch_val_psnr, epoch_val_ssim = val(epoch+1)
    wandb.log({
        "epoch": epoch + 1,
        "train_loss_G": epoch_train_loss.avg,
        "train_loss_D": epoch_train_loss_D.avg,
        "val_psnr": epoch_val_psnr.avg,
        "val_ssim": epoch_val_ssim.avg,
        "learning_rate": optimizer_G.param_groups[0]['lr']
    })
    scheduler.step(epoch_train_loss.avg)
    epoch_end_time = time.time() - epoch_start_time
    
    if epoch%5==0 :
        torch.save(state, os.path.join(saved_weights_dir, 'model_{}.pt'.format(epoch+1)))
        print("Intermediate weights saved")


    if epoch_train_loss.avg < best_loss :
      best_loss = epoch_train_loss.avg
      print("Epoch: {}/{}: Found best loss, weights saved !".format(epoch+1, num_epochs))
      torch.save(state, os.path.join(saved_weights_dir, 'best_model.pt'))

    # Update dictionary
    loss_history["epoch"].append(epoch+1)
    loss_history["lr"].append(optimizer_G.param_groups[0]['lr'])
    loss_history["train loss"].append(epoch_train_loss.avg)

    psnr_metrices_history["epoch"].append(epoch+1)
    psnr_metrices_history["val psnr"].append(epoch_val_psnr.avg)

    ssim_metrices_history["epoch"].append(epoch+1)
    ssim_metrices_history["val ssim"].append(epoch_val_ssim.avg)

    print("\nEpoch: {}/{}: Time taken: {}".format(epoch+1, num_epochs, epoch_end_time))
    print ('Epoch: {}/{}: Train Generator Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_train_loss.avg))
    print ('Epoch: {}/{}: Train Discriminator Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_train_loss_D.avg))

    print ('Epoch: {}/{}: Valid PSNR: {:.4f}'.format(epoch+1, num_epochs, epoch_val_psnr.avg))
    print ('Epoch: {}/{}: Valid SSIM: {:.4f}'.format(epoch+1, num_epochs, epoch_val_ssim.avg))
    print("-"*60)

    # To empty the cache for TQDM
    torch.cuda.empty_cache()
    list(getattr(tqdm, '_instances'))
    for instance in list(tqdm._instances):
        tqdm._decr_instances(instance)
wandb.finish()
total_end_time = time.time() - total_start_time
print("Training completed ! Time taken {}".format(total_end_time))
# Copy output data to CSV files
torch.save(generator.state_dict(), os.path.join(saved_weights_dir, 'full_model.pt'))
