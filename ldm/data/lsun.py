import os, torch, sys, cv2, random, numpy as np
from torch.utils.data import Dataset


class LSUNBase(Dataset):
    def __init__(
        self,
        txt_file,
        data_root,
        size=512,
        interpolation=None,
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
        img = cv2.imread(img_path)
        img = self.crop_and_resize(img)
        img = self.mirror(img)
        img = self.noise(img)
        img = self.blur(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = np.array((img / 255. - 0.5) * 2.).astype(np.float32)
        example = {'image': image}
        return example

    def mirror(self, img):
        if random.random() < self.odds:
            img = cv2.flip(img, 1)
        return img


    def noise(self, img):
        if random.random() < self.odds:
            _noise = np.random.normal(0, 10, img.shape).astype(np.float32)
            img = img.astype(np.float32)
            noisy = cv2.add(img, _noise)
            img = np.clip(noisy, 0, 255).astype(np.uint8)
        return img


    def blur(self, img):                                                                                                                                                                                                
        if random.random() < self.odds:
            img = cv2.GaussianBlur(img, (5, 5), 0)
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

