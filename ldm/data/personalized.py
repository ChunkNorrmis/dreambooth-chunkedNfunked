import os, torch, random
import numpy as np
from typing import OrderedDict
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as fun
from torchvision.io import decode_image
from captionizer import caption_from_path, generic_captions_from_path, find_images


per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

class PersonalizedBase(Dataset):
    def __init__(self, set, reg, data_root, placeholder_token, coarse_class_text, size, repeats, center_crop, flip_p, mixing_prob=0.25, token_only=False, per_image_tokens=False,):

        self.data_root = data_root
        self.imgs = find_images(self.data_root)
        self.n_imgs = len(self.imgs)
        self._length = self.n_imgs
        self.token_only = token_only
        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob
        self.flip_p = flip_p
        self.size = size
        self.repeats = repeats
        self.reg = reg
        self.placeholder_token = placeholder_token
        self.coarse_class_text = coarse_class_text

        if self.reg:
            self.reg_tokens = OrderedDict([('C', self.coarse_class_text)])

        if per_image_tokens:
            assert self.n_imgs < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == 'train':
            self._length = self.n_imgs * self.repeats

    def __len__(self):
        return self._length

    def numpify(self, x): return x.clone().detach().permute(1, 2, 0).cpu().numpy()

    def __getitem__(self, i):
        example = {}
        image = decode_image(self.imgs[i % self.n_imgs], mode='RGB')
        transform = v2.Compose([
            v2.ToDtype(dtype=torch.uint8, scale=True),
            v2.CenterCrop(min(image.size(1), image.size(2))),
            v2.Resize((self.size, self.size), interpolation=3, antialias=True),
            v2.RandomHorizontalFlip(p=self.flip_p),
            v2.GaussianBlur(kernel_size=1, sigma=(0.1, 0.3)),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            v2.Lambda(self.numpify)
        ])
        image = transform(image)
        self.coarse_class_text = self.imgs[i % self.n_imgs].rsplit('/', 2)[1]
        
        if self.reg:
            example['caption'] = generic_captions_from_path(self.imgs[i % self.n_imgs], self.data_root, self.reg_tokens)
        else:
            self.placeholder_token = self.imgs[i % self.n_imgs].rsplit('/', 3)[1]
            example['caption'] = caption_from_path(self.imgs[i % self.n_imgs], self.data_root, self.coarse_class_text, self.placeholder_token)
        example['image'] = image
        
        return example

