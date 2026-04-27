import os, torch, random
import numpy as np
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
        self.repeats = repeats
        self.reg = reg
        self.placeholder_token = placeholder_token
        self.coarse_class_text = coarse_class_text

        if self.reg:
            self.reg_tokens = OrderedDict([('C', self.coarse_class_text)])
            
        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == 'train':
            self._length = self.num_images * self.repeats

        self.transform = v2.Compose([
            v2.ToDtype(dtype=torch.uint8, scale=True),
            v2.Lambda(lambda image: fun.center_crop(image, min(image.shape[1], image.shape[2]))),
            v2.Resize((self.size, self.size), interpolation=3, antialias=True),
            v2.RandomHorizontalFlip(p=self.flip_p),
            v2.GaussianBlur(kernel_size=1, sigma=(0.1, 0.3)),
            v2.Lambda(self.normalize_data)
        ])
        
    def normalize_data(self, image):
        image = image.clone().detach().to(torch.float32).permute(1, 2, 0)
        image = (image / 255 - 0.5) / 0.5
        image = np.array(image)
        return image

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        image_path = self.image_paths[i % self.num_images]
        image = decode_image(image_path, mode='RGB')
        self.coarse_class_text = image_path.rsplit('/', 3)[2]
        self.placeholder_token = image_path.rsplit('/', 3)[1]
        example = {'image': self.transform(image)}
        
        if self.reg:
            example['caption'] = generic_captions_from_path(image_path, self.data_root, self.reg_tokens)
        else:
            example['caption'] = caption_from_path(image_path, self.data_root, self.coarse_class_text, self.placeholder_token)

        return example

        

        
