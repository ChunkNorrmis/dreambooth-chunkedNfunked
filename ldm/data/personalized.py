import os, sys, torch, cv2, random
import numpy as np
from typing import OrderedDict
from torch.utils.data import Dataset
from captionizer import caption_from_path, generic_captions_from_path, find_images



per_img_token_list = ['א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת']

class PersonalizedBase(Dataset):
    def __init__(
        self, data_root=None, set='train', reg=False, placeholder_token='carbuncle', coarse_class_text='_', size=512,
        repeats=100, center_crop=True, flip_p=0.5, mixing_prob=0.25, token_only=False, per_image_tokens=False
    ):
        self.data_root = data_root
        self.imgs = find_images(data_root)
        self.n_imgs = len(self.imgs)
        self._length = self.n_imgs
        self.token_only = token_only
        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.mixing_prob = mixing_prob
        self.size = size
        self.reg = reg
        self.placeholder_token = placeholder_token
        self.coarse_class_text = coarse_class_text
        self.flip_p = flip_p
        
        if per_image_tokens:
            assert self.n_imgs < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."
        if set == 'train':
            self._length = self.n_imgs * repeats
        if self.reg:
            self.reg_tokens = OrderedDict([('C', self.coarse_class_text)])


    def __len__(self):
        return self._length


    def __getitem__(self, i):
        img_path = self.imgs[i % self.n_imgs]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
        img = self.crop_and_resize(img)
        img = self.mirror(img)
        img = random.choice([self.blur, self.sharpen])(img)
        image = img.astype(np.float32)
        image = (image / 255. - 0.5) * 2

        if self.reg:
            caption = generic_captions_from_path(img_path, self.data_root, self.reg_tokens)
        else:
            caption = caption_from_path(img_path, self.data_root, self.coarse_class_text, self.placeholder_token)
        example = {'caption': caption, 'image': img}
        return example


    def mirror(self, img):
        if random.random() < self.flip_p:
            img = cv2.flip(img, 1)
        return img


    def blur(self, img):
        if random.random() < 0.5:
            k = random.randrange(2, 6) * 2 - 1
            img = cv2.GaussianBlur(img, ksize=(k, k), sigmaX=0, sigmaY=0)
        return img


    def sharpen(self, img):
        if random.random() < 0.5:
            blur = cv2.GaussianBlur(img, ksize=(9, 9), sigmaX=10.0, sigmaY=10.0)
            img = cv2.addWeighted(img, alpha=1.5, src2=blur, beta=-0.5, gamma=0.)
            if random.random() < 0.5:
                img = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=0.5, sigmaY=0.5)
        return img


    def crop_and_resize(self, img):
        h, w = img.shape[:2]
        crop = min(h, w)
        if self.center_crop and h != w:
            img = img[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]
        if self.size != crop:
            interp = cv2.INTER_AREA if self.size < crop else cv2.INTER_CUBIC
            img = cv2.resize(img, dsize=(self.size, self.size), interpolation=interp)
        return img

    
