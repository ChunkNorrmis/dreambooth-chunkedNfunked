import os, torch, random
import numpy as np
from PIL import Image
from typing import OrderedDict
from torch.utils.data import Dataset
from torchvision.transforms import v2
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
        self.sz = (self.size, self.size) 
        self.repeats = repeats
        self.reg = reg
        self.placeholder_token = placeholder_token
        self.coarse_class_text = coarse_class_text
        self.normal = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == 'train':
            self._length = self.num_images * self.repeats

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        img_path = self.image_paths[i % self.num_images]
        image = Image.open(img_path)
        transform = v2.Compose([
            v2.RGB(),
            v2.PILToTensor(),
            v2.ToDtype(dtype=torch.uint8, scale=True),
            v2.CenterCrop(min(image.size)),
            v2.Resize((self.size, self.size), interpolation=3, antialias=True),
            v2.RandomHorizontalFlip(p=self.flip_p),
            v2.GaussianBlur(kernel_size=1, sigma=(0.1, 0.3)),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=self.normal, std=self.normal),
            v2.Lambda(lambda x : x.clone().detach().permute(1, 2, 0).numpy())
        ])
        image = transform(image)
        img_class = img_path.rsplit('/')[1]
        example = {'image': image}
        if self.reg:
            reg_tokens = OrderedDict([('C', img_class)])
            example["caption"] = generic_captions_from_path(img_path, self.data_root, reg_tokens)
        else:
            img_token = img_path.rsplit('/')[2]
            example["caption"] = caption_from_path(img_path, self.data_root, img_class, img_token)
        
        return example

