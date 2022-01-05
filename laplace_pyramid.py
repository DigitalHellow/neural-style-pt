"""
Notation from http://sepwww.stanford.edu/data/media/public/sep/morgan/texturematch/paper_html/node3.html
"""

# %%
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import cv2

from typing import List, Tuple
from PIL import Image


_KERNEL_SIZE = (7, 7)
_INTER_TYPE = transforms.InterpolationMode.NEAREST


def set_kernel_size(kernel_size: Tuple[int, int]) -> \
        Tuple[bool, str]:
    global _KERNEL_SIZE
    
    if len(kernel_size) != 2:
        return (False, "Invalid kernel length. length must be 2")

    if type(kernel_size) not in [tuple, list]:
        return (False, "Invalid kernel type. Valid types are tuple and list")

    if kernel_size[0] % 2 or kernel_size[1] % 2:
        return (False, "Invalid kernel value. Values must be odd")

    _KERNEL_SIZE = kernel_size
    return True


def pyramid_down(img: torch.Tensor, n_scales: int=1) -> \
        Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Performs blur and downsample.

    """
    low = []
    high = []
    l_minus = img
    for i in range(n_scales):
        # low pass
        l_i = F.gaussian_blur(l_minus, _KERNEL_SIZE)
        low.append(l_i)

        # high pass
        h_i = (l_i - l_minus)
        high.append(h_i)

        if i != n_scales - 1:
            l_minus = transforms.Resize(l_i.size()[-1] // 2,
                interpolation=_INTER_TYPE)(l_i)

    return (low, high)


def pyramid_up(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs blur and upsample
    """


# %%
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    set_kernel_size((15, 15))

    img = Image.open("examples/inputs/golden_gate.jpg")
    convert_tensor = transforms.ToTensor()
    img = convert_tensor(img)
    low, high = pyramid_down(img, 3)

    for i in range(len(low)):
        fig, arr = plt.subplots(1, 2)
        arr[0].imshow(low[i].transpose(-1,1).numpy().T)
        arr[1].imshow(high[i].transpose(-1,1).numpy().T)
        plt.show()
# %%
