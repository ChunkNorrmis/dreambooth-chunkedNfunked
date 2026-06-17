import os, torch, sys, cv2, random, numpy as np
from torch.utils.data import Dataset


class LSUNBase(Dataset):
    def __init__(
        self,
        txt_file,
        data_root,
        size=512,
        interpolation="bicubic",
        flip_p=0.5
    ):
        self.data_paths = txt_file
        self.data_root = data_root
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.size = size
        self.flip_p = flip_p
        self.center_crop = False
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l) for l in self.image_paths]
        }


    def __len__(self):
        return self._length


    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        img_path = example["file_path_"]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
        image = self.crop_and_resize(image)
        image = self.mirror(image)
        image = random.choice([self.blur, self.sharpen])(image)
        image = np.array(image).astype(np.float32)
        image = (image  / 255. - 0.5) * 2
        return {'image': image}
        

    def mirror(self, img):
        if random.random() < self.flip_p:
            img = cv2.flip(img, 1)
        return img


    def blur(self, img):
        if random.random() < 0.5:
            r = [n / 10 for n in range(6, 11)] + [0]
            sig = random.choice(r)
            img = cv2.GaussianBlur(img, ksize=(3, 3), sigmaX=sig, sigmaY=sig)
        return img


    def sharpen(self, img):
        if random.random() < 0.5:
            mask = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=0, sigmaY=0)
            alpha = 1.3
            beta = 1 - alpha
            sharpened = cv2.addWeighted(img, alpha=alpha, src2=mask, beta=beta, gamma=0.0)
            img = cv2.GaussianBlur(sharpened, ksize=(3, 3), sigmaX=0.5, sigmaY=0.5)
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


class LSUNChurchesTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/church_outdoor_train.txt", data_root="data/lsun/churches", **kwargs)


class LSUNChurchesValidation(LSUNBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/lsun/church_outdoor_val.txt", data_root="data/lsun/churches",
                         flip_p=flip_p, **kwargs)


class LSUNBedroomsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/bedrooms_train.txt", data_root="data/lsun/bedrooms", **kwargs)


class LSUNBedroomsValidation(LSUNBase):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(txt_file="data/lsun/bedrooms_val.txt", data_root="data/lsun/bedrooms",
                         flip_p=flip_p, **kwargs)


class LSUNCatsTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="data/lsun/cat_train.txt", data_root="data/lsun/cats", **kwargs)


class LSUNCatsValidation(LSUNBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="data/lsun/cat_val.txt", data_root="data/lsun/cats",
                         flip_p=flip_p, **kwargs)

