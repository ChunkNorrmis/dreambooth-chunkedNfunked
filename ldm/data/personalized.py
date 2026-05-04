import os, torch, random, cv2
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
        data_root,
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

        if per_image_tokens:
            assert self.n_imgs < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == 'train':
            self._length = self.n_imgs * self.repeats

        if self.reg:
            self.reg_tokens = OrderedDict([('C', self.coarse_class_text)])

    def __len__(self):
        return self._length

    def transform(self, x):
        x = cv2.imread(x)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        h, w = x.shape[0], x.shape[1]
        crop = min(h, w)
        if self.center_crop and h != w:
            x = x[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]
        if crop > self.size:
            x = cv2.resize(x, dsize=(self.size, self.size), interpolation=cv2.INTER_AREA)
        if random.random() < self.flip_p:
            x = cv2.flip(x, 1)
        x = (x.astype(np.float32) / 255 - 0.5) / 0.5
        return x

    def __getitem__(self, i):
        example = {}
        img = self.imgs[i % self.n_imgs]
        
        if self.reg:
            example['caption'] = generic_captions_from_path(self.imgs[i % self.n_imgs], self.data_root, self.reg_tokens)
        else:
            example['caption'] = caption_from_path(self.imgs[i % self.n_imgs], self.data_root, self.coarse_class_text, self.placeholder_token)
        example['image'] = self.transform(img)
        return example

