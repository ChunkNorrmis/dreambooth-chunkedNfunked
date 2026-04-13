import os, torch, random
import numpy as np
from typing import OrderedDict
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
from captionizer import caption_from_path, generic_captions_from_path, find_images


per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

class PersonalizedBase(Dataset):
    def __init__(self, set, data_root, size, repeats, placeholder_token, coarse_class_text, center_crop,
                 mixing_prob, flip_p=0.5, reg=False, token_only=False, per_image_tokens=False,):

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
        self.flip_p = flip_p
        self.coarse_class_text = coarse_class_text
        self.size = size
        self.repeats = repeats
        self.reg = reg

        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * self.repeats

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        img_path = self.image_paths[i % self.num_images]
        image = Image.open(img_path)
        crop = min(image.size)
        transform = v2.Compose([
            v2.RGB(),
            v2.PILToTensor(),
            v2.ToDtype(dtype=torch.uint8, scale=True),
            v2.CenterCrop((crop, crop)),
            v2.Resize((self.size, self.size), interpolation=3, antialias=True),
            v2.RandomHorizontalFlip(p=self.flip_p),
            v2.GaussianBlur(kernel_size=1, sigma=0.2),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Lambda(lambda x: ((x.clone().detach() - 0.5) / 0.5).permute(1, 2, 0).cpu().numpy())
        ])
        example = {'image': transform(image)}
        class_dir = os.path.dirname(img_path)
        im_class = os.path.basename(class_dir)
        if self.reg:
            reg_tokens = OrderedDict([('C', im_class)])
            example["caption"] = generic_captions_from_path(image, self.data_root, reg_tokens)
        else:
            token_dir = os.path.dirname(class_dir)
            im_token = os.path.basename(token_dir)
            example["caption"] = caption_from_path(image, self.data_root, im_class, im_token)
                
        return example

