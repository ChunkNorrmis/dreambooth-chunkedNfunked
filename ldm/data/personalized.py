import os, torch, random
import numpy as np
from typing import OrderedDict
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as fun
from PIL import Image
from captionizer import caption_from_path, generic_captions_from_path, find_images


per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

class PersonalizedBase(Dataset):
    def __init__(
        self,
        set='train',
        reg=False,
        data_root=None,
        placeholder_token='rock',
        coarse_class_text='lobster',
        size=512,
        repeats=100,
        center_crop=False,
        flip_p=0.5,
        mixing_prob=0.25,
        token_only=False,
        per_image_tokens=False
    ):
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
        self.nml = torch.tensor([0.5,0.5,0.5])

        if per_image_tokens:
            assert self.n_imgs < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == 'train':
            self._length = self.n_imgs * self.repeats

        if self.reg:
            self.reg_tokens = OrderedDict([('C', self.coarse_class_text)])


    def __len__(self):
        return self._length


    def numpify(self, x): return x.permute(2,1,0).permute(1,0,2).cpu().numpy(force=True)


    def crop_and_resize(self, x):
        h, w = x.size(1), x.size(2)
        crop = min(h, w)
        if self.center_crop and x.shape[1] != x.shape[2]:
            x = fun.center_crop(x, crop)
        if self.size < crop:
            x = fun.resize(x, size=(self.size, self.size), interpolation=3, antialias=True)
        return x


    def __getitem__(self, i):
        example = {}
        image = Image.open(self.imgs[i % self.n_imgs])
        transform = v2.Compose([
            v2.RGB(),
            v2.PILToTensor(),
            v2.ToDtype(dtype=torch.uint8, scale=True),
            v2.Lambda(self.crop_and_resize),
            v2.RandomHorizontalFlip(p=self.flip_p),
            #v2.GaussianBlur(kernel_size=1, sigma=0.2),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=self.nml, std=self.nml),
            v2.Lambda(self.numpify)
        ])

        if self.reg:
            example['caption'] = generic_captions_from_path(self.imgs[i % self.n_imgs], self.data_root, self.reg_tokens)
        else:
            example['caption'] = caption_from_path(self.imgs[i % self.n_imgs], self.data_root, self.coarse_class_text, self.placeholder_token)
        
        example['image'] = transform(image)
        return example



