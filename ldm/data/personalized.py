import os, torch, cv2
from random import random
import numpy as np
from typing import OrderedDict
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as fun
from captionizer import caption_from_path, generic_captions_from_path, find_images


per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

class PersonalizedBase(Dataset):
    def __init__(
        self,
        data_root=None,
        set='train',
        reg=False,
        placeholder_token='lobster',
        coarse_class_text=None,
        size=512,
        repeats=100,
        center_crop=False,
        flip_p=0.5,
        mixing_prob=0.25,
        token_only=False,
        per_image_tokens=False
    ):
        self.set = set
        self.data_root = data_root
        self.imgs = find_images(self.data_root)
        self.n_imgs = len(self.imgs)
        self._length = self.n_imgs
        self.token_only = token_only
        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob
        self.size = size
        self.reg = reg
        self.placeholder_token = placeholder_token
        self.coarse_class_text = coarse_class_text
        self.tt_nml = torch.tensor([0.5,0.5,0.5], dtype=torch.float32)
        self.np_nml = np.array([0.5,0.5,0.5], dtype=np.float32)
        self.scl = np.array([255,255,255], dtype=np.float32)
        self.flip_p = flip_p
            
        if per_image_tokens:
            assert self.n_imgs < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if self.set == 'train':
            self._length = self.n_imgs * repeats

        if self.reg:
            self.flip_p = 0.0
            self.reg_tokens = OrderedDict([('C', self.coarse_class_text)])

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        img_path = self.imgs[i % self.n_imgs]
        image = self.transformer(img_path)
        example['image'] = image
        if self.reg:
            example['caption'] = generic_captions_from_path(img_path, self.data_root, self.reg_tokens)
        else:
            example['caption'] = caption_from_path(img_path, self.data_root, self.coarse_class_text, self.placeholder_token)
        return example


    def transformer(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[0], image.shape[1]
        crop = min(h, w)
        if self.center_crop and h != w:
            image = image[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]
        if self.size != crop:
            interp = cv2.INTER_AREA if self.size < crop else cv2.INTER_CUBIC
            image = cv2.resize(image, dsize=(self.size, self.size), interpolation=interp)
        if self.flip_p > random():
            image = cv2.flip(image, 1)
        return (image.astype(np.float32) / self.scl - self.np_nml) / self.np_nml

    def transforms(self, img_path):
        image = decode_image(img_path, mode='RGB')
        image = image.to(torch.device('cuda:0'))
        transform = v2.Compose([
            v2.ToDtype(dtype=torch.uint8, scale=True),
            v2.Lambda(self.crop_and_resize),
            v2.RandomHorizontalFlip(p=self.flip_p),
            #v2.GaussianBlur(kernel_size=1, sigma=0.2),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=self.tt_nml, std=self.tt_nml),
            v2.Lambda(self.numpify)
        ])
        return transform(image)
    
    def numpify(self, x): return x.permute(2,1,0).permute(1,0,2).cpu().numpy(force=True).astype(np.float32)

    def crop_and_resize(self, x):
        h, w = x.size(1), x.size(2)
        crop = min(h, w)
        if self.center_crop and x.shape[1] != x.shape[2]:
            x = fun.center_crop(x, crop)
        if self.size != crop:
            x = fun.resize(x, size=(self.size, self.size), interpolation=3, antialias=True)
        return x

