import os, torch, random, sys
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import v2
import glob
from PIL import Image

imagenet_templates_small = [
    'a painting in the style of {}',
    'a rendering in the style of {}',
    'a cropped painting in the style of {}',
    'the painting in the style of {}',
    'a clean painting in the style of {}',
    'a dirty painting in the style of {}',
    'a dark painting in the style of {}',
    'a picture in the style of {}',
    'a cool painting in the style of {}',
    'a close-up painting in the style of {}',
    'a bright painting in the style of {}',
    'a cropped painting in the style of {}',
    'a good painting in the style of {}',
    'a close-up painting in the style of {}',
    'a rendition in the style of {}',
    'a nice painting in the style of {}',
    'a small painting in the style of {}',
    'a weird painting in the style of {}',
    'a large painting in the style of {}',
]

imagenet_dual_templates_small = [
    'a painting in the style of {} with {}',
    'a rendering in the style of {} with {}',
    'a cropped painting in the style of {} with {}',
    'the painting in the style of {} with {}',
    'a clean painting in the style of {} with {}',
    'a dirty painting in the style of {} with {}',
    'a dark painting in the style of {} with {}',
    'a cool painting in the style of {} with {}',
    'a close-up painting in the style of {} with {}',
    'a bright painting in the style of {} with {}',
    'a cropped painting in the style of {} with {}',
    'a good painting in the style of {} with {}',
    'a painting of one {} in the style of {}',
    'a nice painting in the style of {} with {}',
    'a small painting in the style of {} with {}',
    'a weird painting in the style of {} with {}',
    'a large painting in the style of {} with {}',
]

per_img_token_list = [
    'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת',
]

class PersonalizedBase(Dataset):
    def __init__(
        self,
        set,
        data_root,
        size,
        repeats,
        flip_p,
        placeholder_token,
        per_image_tokens,
        center_crop
    ):
        super().__init__()
        self.data_root = data_root
        self.image_paths = [os.path.relpath(im) for im in glob.glob(os.path.join(self.data_root, '**', '*.png'), recursive=True)]
        self.num_images = len(self.image_paths)
        self._length = self.num_images
        self.flip_p = flip_p
        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.size = size
        self.repeats = repeats
        self.placeholder_token = placeholder_token
        
                        
        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * self.repeats

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image_path = self.image_paths[i % self.num_images]
        image = decode_image(image_path, mode='RGB')
        crop = min(image.size(1), image.size(2))
        transform = v2.Compose([
            v2.ToDtype(dtype=torch.uint8, scale=True),
            v2.CenterCrop((crop, crop)),
            v2.Resize((self.size, self.size), interpolation=3, antialias=True),
            v2.RandomHorizontalFlip(p=self.flip_p),
            v2.GaussianBlur(kernel_size=1, sigma=(0.1, 0.3)),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=[0.5], std=[0.5]),
            v2.Lambda(lambda img : img.detach().permute(1, 2, 0).numpy())
        ])
        example['image'] = transform(image)}
        self.coarse_class_text = image_path.rsplit('/', 3)[2]
        self.placeholder_token = image_path.rsplit('/', 3)[1]
        
        if self.per_image_tokens and random.random() < 0.25:
            example['caption'] = random.choice(imagenet_dual_templates_small).format(self.placeholder_token, per_img_token_list[i % self.num_images])
        else:
            example['caption'] = random.choice(imagenet_templates_small).format(self.placeholder_token)
        
        return example




