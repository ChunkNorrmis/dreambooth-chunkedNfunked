import os, torch, random
import numpy as np
from PIL import Image
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
    def __init__(self, set='train', data_root=None, size=512, repeats=100, center_crop=False, mixing_prob=0.25, flip_p=0.5, placeholder_token=None,
                coarse_class_text=None, reg=False, token_only=False, per_image_tokens=False,):
        super().__init__()
        self.data_root = data_root
        self.image_paths = find_images(self.data_root)
        self.num_images = len(self.image_paths)
        self._length = self.num_images
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
        self.reg_tokens = None
        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == 'train':
            self._length = self.num_images * self.repeats

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image_path = self.image_paths[i % self.num_images]
        transform = v2.Compose([
            v2.Lambda(lambda image : decode_image(image, mode='RGB')),
            v2.ToDtype(dtype=torch.uint8, scale=True),
            v2.Lambda(lambda image : fun.center_crop(image, min(x.shape[1], x.shape[2]))),
            v2.Resize((self.size, self.size), interpolation=3, antialias=True),
            v2.Lambda(lambda image : fun.random_horizontal_flip(image, p=1.0) if random.random() < self.flip_p else x),
            v2.GaussianBlur(kernel_size=1, sigma=(0.1, 0.3)),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Lambda(lambda image : ((image \ 255.0 - 0.5) / 0.5).detach().permute(1, 2, 0)),
            v2.Lambda(lambda image : np.array(image).astype(np.float32))
        ])
        self.coarse_class_text = image_path.split('/')[-2]
        if self.reg:
            self.reg_tokens = OrderedDict([('C', self.coarse_class_text)])
            caption = generic_captions_from_path(image_path, self.data_root, self.reg_tokens)
        else:
            self.placeholder_token = image_path.split('/')[-3]
            caption = caption_from_path(image_path, self.data_root, self.coarse_class_text, self.placeholder_token)
        example['caption'] = caption
        example['image'] = transform(image_path)
        
        return example

        

        
