import os, torch, random, sys, cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import decode_image
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as fun
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
        data_root=None,
        set='train',
        size=512,
        repeats=100,
        flip_p=0.5,
        placeholder_token='rock',
        per_image_tokens=False,
        center_crop=False
    ):
        super().__init__()
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
    
    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        img_path = self.imgs[i % self.n_imgs]
        image = self.transformer(img_path)
        
        if self.per_image_tokens and random.random() < 0.25:
            example['caption'] = random.choice(imagenet_dual_templates_small).format(self.placeholder_token, per_img_token_list[i % self.n_imgs])
        else:
            example['caption'] = random.choice(imagenet_templates_small).format(self.placeholder_token)
        
        example['image'] = image
        return example


    def transformer(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[0], image.shape[1]
        crop = min(h, w)
        if self.center_crop and h != w:
            image = image[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]
        if self.size != crop:
            interp = cv2.INTER_AREA if self.size < crop else cv2.INTER_CUBIC
            image = cv2.resize(image, dsize=(self.size, self.size), interpolation=interp)
        if self.flip_p > random():
            image = cv2.flip(image, 1)
        image = cv2.GaussianBlur(image, ksize=3, sigmaX=0.2, sigmaY=0.2)
        image = ((image / 255. - 0.5) / 0.5).astype(np.float32)
        return image


    def transforms(self, img_path):
        img = decode_image(img_path, mode='RGB')
        img = img.to(torch.device('cuda'))
        transform = v2.Compose([
            v2.Lambda(self.crop_and_resize),
            v2.RandomHorizontalFlip(p=self.flip_p),
            v2.GaussianBlur(kernel_size=1, sigma=0.3),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=[0.5], std=[0.5]),
            v2.Lambda(self.numpify)
        ])
        image = transform(img)
        return image

    def numpify(self, img): return img.permute(2,1,0).permute(1,0,2).cpu().numpy(force=True).astype(np.float32)

    def crop_and_resize(self, img):
        h, w = img.size(1), img.size(2)
        crop = min(h, w)
        if self.center_crop and img.shape[1] != img.shape[2]:
            img = fun.center_crop(img, crop)
        if self.size != crop:
            img = fun.resize(img, size=(self.size, self.size), interpolation=3, antialias=True)
        return img


