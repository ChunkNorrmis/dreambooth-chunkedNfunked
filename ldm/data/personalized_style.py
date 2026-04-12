import os, torch, random, sys
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2

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
        self.image_paths = os.listdir(self.data_root)
        self.num_images = len(self.image_paths)
        self._length = self.num_images
        self.placeholder_token = placeholder_token
        self.chance = flip_p
        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.size = size
                        
        if per_image_tokens:
            assert self.num_images < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."

        if set == "train":
            self._length = self.num_images * repeats

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
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
            v2.Lambda(self.normpy)
        ])
        example['image'] = transform(image)
        class_dir = os.path.dirname(img_path)
        im_class = os.path.basename(class_dir)
        token_dir = os.path.dirname(class_dir)
        im_token = os.path.basename(token_dir)
        if self.per_image_tokens and random.random() < 0.25:
            example["caption"] = random.choice(imagenet_dual_templates_small).format(im_token, per_img_token_list[i % self.num_images])
        else:
            example["caption"] = random.choice(imagenet_templates_small).format(im_token)

        return example

    def normpy(self, x):
        x = x.clone().detach().permute(1, 2, 0).numpy()
        x = np.array(x).astype(np.float32)
        return (x / 255 - 0.5) / 0.5

