import os, torch, sys, cv2, random, numpy as np
from torch.utils.data import Dataset
import glob

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
        center_crop=True,
        mixing_prob=0.25
    ):
        self.data_root = data_root
        self.imgs = [os.path.relpath(im) for im in glob.glob(os.path.join(self.data_root, '**', '*.png'), recursive=True)]
        self.n_imgs = len(self.imgs)
        self._length = self.n_imgs
        self.flip_p = flip_p
        self.per_image_tokens = per_image_tokens
        self.center_crop = center_crop
        self.size = size
        self.placeholder_token = placeholder_token
        self.mixing_prob = mixing_prob
        if per_image_tokens:
            assert self.n_imgs < len(per_img_token_list), f"Can't use per-image tokens when the training set contains more than {len(per_img_token_list)} tokens. To enable larger sets, add more tokens to 'per_img_token_list'."
        if set == "train":
            self._length = self.n_imgs * repeats


    def __len__(self):
        return self._length


    def __getitem__(self, i):
        example = {}
        img_path = self.imgs[i % self.n_imgs]
        image = self.augment(img_path)
        if self.per_image_tokens and random.random() < self.mixing_prob:
            caption = random.choice(imagenet_dual_templates_small).format(self.placeholder_token, per_img_token_list[i % self.n_imgs])
        else:
            caption = random.choice(imagenet_templates_small).format(self.placeholder_token)
        example = {'caption': caption, 'image': image}
        return example


    def augment(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.crop_and_resize(img)
        img = self.random_mirror(img)
        img = self.random_blur(img)
        return np.array(((img / 255. - 0.5) * 2.), dtype=np.float32)


    def random_mirror(self, img):
        if random.random() < self.flip_p:
            img = cv2.flip(img, 1)
        return img


    def random_blur(self, img):
        if random.random() < 0.5:
            k = random.choice([1, 3, 1])
            x = random.uniform(0.2, 0.5)
            img = cv2.GaussianBlur(img, ksize=(k,k), sigmaX=x, sigmaY=x)
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



