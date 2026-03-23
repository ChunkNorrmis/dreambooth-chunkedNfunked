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
    def __init__(self, set, reg, data_root, size, repeats, flip_p, placeholder_token, coarse_class_text,
                token_only, per_image_tokens, center_crop, mixing_prob):
        
        super().__init__()
        self.data_root = data_root
        self.image_paths = find_images(self.data_root)
        self.num_images = len(self.image_paths)
        self._length = self.num_images
        self.placeholder_token = placeholder_token
        self.token_only = token_only
        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob
        self.chance = flip_p
        self.coarse_class_text = coarse_class_text
        self.size = size
        self.reg = reg
        self.repeats = repeats
                       
        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * self.repeats
        
        if self.reg and self.coarse_class_text:
            self.reg_tokens = OrderedDict([('C', self.coarse_class_text)])
    
    def to_nda(self, img):
        img = np.array(img.detach().permute(1, 2, 0)).astype(np.uint8)
        img = np.array(img / 127.5 - 1.0).astype(np.float32)
        return img
    
    def g_blur(self, img):
        if self.chance > random.random():
            img = fun.gaussian_blur(img, kernel_size=1, sigma=(0.1, 0.5))
        return img
    
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image_path = self.image_paths[i % self.num_images]
        img = decode_image(image_path, mode="RGB")
        transform = v2.Compose([
            v2.ToDtype(dtype=torch.uint8, scale=True),
            v2.RandomCrop(min(img.shape[1], img.shape[2])),
            v2.Resize((self.size, self.size), interpolation=3, antialias=True),
            v2.RandomAdjustSharpness(sharpness_factor=random.uniform(1.1, 1.7), p=self.chance),
            v2.Lambda(self.g_blur),
            v2.RandomHorizontalFlip(p=self.chance),
            v2.RandomPerspective(distortion_scale=0.25, p=self.chance, interpolation=2),
            v2.Lambda(self.to_nda)
        ])
        example['image'] = transform(img)
        
        if self.reg and self.coarse_class_text:
            example["caption"] = generic_captions_from_path(image_path, self.data_root, self.reg_tokens)
        else:
            example["caption"] = caption_from_path(image_path, self.data_root, self.coarse_class_text, self.placeholder_token)
        
        return example

