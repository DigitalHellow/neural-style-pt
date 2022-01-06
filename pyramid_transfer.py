#%%
import os
import time
import numpy as np
import torch
from torch.optim import Adam
from torchvision import transforms

import matplotlib.pyplot as plt

from PIL import Image
from typing import Tuple

import laplace_pyramid
from transformer_net import TransformerNet

def main(target_path: str, style_pth: str, image_size: int,
        lr: float, style_weight: float, target_weight: float,
        device: str, epochs: int, iter_n: int,
        save_dir: str) -> None:

    style_img = load_image(style_pth, size=image_size)
    
    style_transform = transforms.ToTensor()

    target_img = load_image(target_path, size=image_size)

    low_features, _ = laplace_pyramid.pyramid_down(
        style_transform(style_img), 3)
    
    _, high_features = laplace_pyramid.pyramid_down(
        style_transform(target_img), 3)
    
    for i in range(len(high_features)):
        high_features[i] = high_features[i].to(device)

    target_img_t = style_transform(target_img)[None, ...]
    print(target_img_t.size())

    # gram style for style image
    
    style_gram = gram_matrix(low_features[-1]).to(device)
    
    # init network and loss
    transformer = TransformerNet().to(device)
    
    opt = Adam(transformer.parameters(), lr)
    mse_loss = torch.nn.MSELoss()

    for epoch in range(epochs):
        epoch_start = time.time()
        for i in range(iter_n):
            opt.zero_grad()
            synth_img = transformer(target_img_t)
            synth_lowf, synth_highf = laplace_pyramid.pyramid_down(
                synth_img, 3)
            
            target_loss = target_weight * mse_loss(
                high_features[-1][None, ...], synth_highf[-1])
            
            style_loss = style_weight * mse_loss(
                gram_matrix(synth_lowf[0][0]), style_gram)

            total_loss = target_loss + style_loss
            total_loss.backward()

            opt.step()

            # debugging stuff
            if i % 100 == 0:
                mesg = f"Total loss: {total_loss}, style loss: {style_loss}"
                mesg += f", target loss {target_loss}\n"
                mesg += f"Time since epoch {epoch} started: {time.time() - epoch_start}\n"
                print(mesg)
                plt.imshow(synth_img[0].transpose(-1,1).detach().numpy().T)
                plt.show()


    transformer.eval().cpu()
    save_model_filename = "epoch_" + str(epochs) + "_" + str(time.ctime()).replace(' ', '_') + "_" + str(
         target_weight) + "_" + str(style_weight) + ".model"
    save_model_path = os.path.join(save_dir, save_model_filename)
    # torch.save(transformer.state_dict(), save_model_path)

    print("\nDone, trained model saved at", save_model_path)
    


def remove_transparency(im: Image) -> Image:
    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        return im.convert('RGB')
    else:
        return im


def load_image(filename: str, size: int=None, scale=None) -> Image:
    img = remove_transparency(Image.open(filename))
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_image(filename: str, data: torch.Tensor) -> None:
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


def gram_matrix(y: torch.Tensor):
    (ch, h, w) = y.size()
    features = y.view(ch, w * h)
    features_t = features.transpose(0, 1)
    gram = features.mm(features_t) / (ch * h * w)
    return gram



if __name__ == "__main__":
    target_path = "examples/inputs/golden_gate.jpg"
    style_pth = "examples/inputs/starry_night_google.jpg"
    image_size = 512
    lr = 1e-3
    target_weight = 1e5
    style_weight = 1e6
    iter_n = 500
    epochs = 1
    device = "cpu"
    save_dir = "models/"

    main(target_path, style_pth, image_size,
        lr, style_weight, target_weight,
        device, epochs, iter_n,
        save_dir)
# %%
