import os, torch, cv2
from random import random
import numpy as np
from typing import OrderedDict
from torch.utils.data import Dataset
from torchvision.io import decode_image
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
        image = self.transform(img_path)
        
        if self.reg:
            example['caption'] = generic_captions_from_path(img_path, self.data_root, self.reg_tokens)
        else:
            example['caption'] = caption_from_path(img_path, self.data_root, self.coarse_class_text, self.placeholder_token)
        
        example['image'] = image
        
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
        image = (image.astype(np.float32) / 255 - 0.5) / 0.5
        return image


    def transforms(self, img_path):
        transform = v2.Compose([
            v2.Lambda(lambda x: decode_image(x, mode='RGB')),
            v2.Lambda(lambda x: x.to(torch.device('cuda:0'))),
            v2.Lambda(self.crop_and_resize),
            v2.RandomHorizontalFlip(p=self.flip_p),
            #v2.GaussianBlur(kernel_size=1, sigma=0.2),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=[0.5], std=[0.5]),
            v2.Lambda(self.numpify)
        ])
        image = transform(img_path)
        return image

    def numpify(self, image): return image.permute(2,1,0).permute(1,0,2).cpu().numpy(force=True).astype(np.float32)

    def crop_and_resize(self, image):
        h, w = image.size(1), image.size(2)
        crop = min(h, w)
        if self.center_crop and image.shape[1] != image.shape[2]:
            image = fun.center_crop(image, crop)
        if self.size != crop:
            image = fun.resize(image, size=(self.size, self.size), interpolation=3, antialias=True)
        return image




