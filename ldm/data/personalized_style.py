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

        self.data_root = data_root
        self.imgs = [os.path.relpath(im) for im in glob.glob(os.path.join(self.data_root, '**', '*.png'), recursive=True)]
        self.n_imgs = len(self.imgs)
        self._length = self.n_imgs
        self.flip_p = flip_p
        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.size = size
        self.repeats = repeats
        self.placeholder_token = placeholder_token
        
                        
        if per_image_tokens:
            assert self.n_imgs < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.n_imgs * self.repeats

    def normpy(self, t):
        t = t.detach().to(torch.float32)
        t = (t / 255 - 0.5) / 0.5
        n = np.array(t.permute(1, 2, 0))
        return n
    
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = decode_image(self.imgs[i % self.n_imgs], mode='RGB')
        transform = v2.Compose([
            v2.ToDtype(dtype=torch.uint8, scale=True),
            v2.CenterCrop(min(image.size(1), image.size(2))),
            v2.Resize((self.size, self.size), interpolation=3, antialias=True),
            v2.RandomHorizontalFlip(p=self.flip_p),
            v2.GaussianBlur(kernel_size=1, sigma=(0.1, 0.3)),
            v2.Lambda(self.normpy)
        ])
        example['image'] = transform(image)
        self.coarse_class_text = self.imgs[i % self.n_imgs].rsplit('/', 3)[2]
        self.placeholder_token = self.imgs[i % self.n_imgs].rsplit('/', 3)[1]
        
        if self.per_image_tokens and random.random() < 0.25:
            example['caption'] = random.choice(imagenet_dual_templates_small).format(self.placeholder_token, per_img_token_list[i % self.n_imgs])
        else:
            example['caption'] = random.choice(imagenet_templates_small).format(self.placeholder_token)
        
        return example




