import os, sys, torch, cv2, random, glob, numpy as np
from typing import OrderedDict
from torch.utils.data import Dataset
from captionizer import caption_from_path, generic_captions_from_path, find_images
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as fun


per_img_token_list = ['א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'כ', 'ל', 'מ', 'נ', 'ס', 'ע', 'פ', 'צ', 'ק', 'ר', 'ש', 'ת']

class PersonalizedBase(Dataset):
    def __init__(self, data_root=None, set='train', reg=False, placeholder_token='lobster', coarse_class_text=None,
                 size=512, epochs=100, center_crop=True, flip_p=0.5, mixing_prob=0.25, token_only=False, per_image_tokens=False):

        self.data_root = data_root
        self.imgs = [os.path.relpath(im) for im in glob.glob(os.path.join(self.data_root, '**', '*.png'), recursive=True)]
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
            self._length = self.n_imgs * epochs
        if self.reg:
            self.reg_tokens = OrderedDict([('C', self.coarse_class_text)])


    def __len__(self):
        return self._length

    def __getitem__(self, i):
        img_path = self.imgs[i % self.n_imgs]
        img = cv2.imread(img_path)
        img = self.crop_and_resize(img)
        img = self.mirror(img)
        img = self.noise(img)
        img = self.blur(img)
        image = self.convert(img)
        
        if self.reg:
            caption = generic_captions_from_path(img_path, self.data_root, self.reg_tokens)
        else:
            caption = caption_from_path(img_path, self.data_root, self.coarse_class_text, self.placeholder_token)
        example = {'caption': caption, 'image': image}
        return example
        

    def convert(self, img):
        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0)
        if isinstance(img, np.ndarray):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = np.array((img / 255. - 0.5) * 2).astype(np.float32)
        return image


    def mirror(self, img):
        if random.random() < self.flip_p:
            img = cv2.flip(img, 1)
        return img


    def noise(self, img):
        if random.random() < 0.25:
            if isinstance(img, torch.Tensor):
                img = self.from_tensor(img)
            _noise = np.array(img, dtype=np.uint8)
            cv2.randn(_noise, 0, 25)
            img = cv2.add(img, _noise)
        return img


    def posterize(self, img):
        if random.random() < 0.5:
            if isinstance(img, np.ndarray):
                img = self.to_tensor(img)
            img = fun.posterize(img, bits=2)
        return img


    def saturation(self, img):
        if random.random() < 0.25:
            if isinstance(img, np.ndarray):
                img = self.to_tensor(img)
            img = fun.adjust_saturation(img, saturation_factor=1.15)
        return img


    def brightness(self, img):
        if random.random() < 0.25:
            if isinstance(img, np.ndarray):
                img = self.to_tensor(img)
            img = fun.adjust_brightness(img, brightness_factor=1.15)
        return img


    def contrast(self, img):
        if random.random() < 0.25:
            if isinstance(img, np.ndarray):
                img = self.to_tensor(img)
            img = fun.adjust_contrast(img, contrast_factor=1.15)
        return img


    def to_tensor(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img, dtype=torch.uint8).permute(2, 0, 1)
        return img


    def from_tensor(self, img):
        img = np.array(img, dtype=np.uint8).transpose(1, 2, 0)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img


    def blur(self, img):
        if random.random() < 0.25:
            if isinstance(img, torch.Tensor):
                img = self.from_tensor(img)
            img = cv2.GaussianBlur(img, (3, 3), 0)
        return img


    def crop_and_resize(self, img):
        h, w = img.shape[:2]
        crop = min(h, w)
        if self.center_crop and h != w:
            img = img[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]
        if self.size != crop:
            interp = cv2.INTER_AREA if self.size < crop else cv2.INTER_CUBIC
            img = cv2.resize(img, (self.size, self.size), interp)
        return img

