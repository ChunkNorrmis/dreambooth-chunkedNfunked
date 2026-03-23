import os, torch, random
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as fun


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
        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]
        self.num_images = len(self.image_paths)
        self._length = self.num_images
        self.placeholder_token = placeholder_token
        self.flip_p = flip_p
        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.size = size
        self.repeats = repeats
                        
        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * self.repeats

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
        
        if self.per_image_tokens and np.random.uniform() < 0.25:
            example["caption"] = random.choice(imagenet_dual_templates_small).format(self.placeholder_token, per_img_token_list[i % self.num_images])
        else:
            example["caption"] = random.choice(imagenet_templates_small).format(self.placeholder_token)

        return example

