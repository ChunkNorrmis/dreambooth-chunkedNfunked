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
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l) for l in self.image_paths]
        }


    def __len__(self):
        return self._length


    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        img_path = example["file_path_"]
        image = self.transform(img_path)
        example['image'] = image
        return example


    def transform(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[0], image.shape[1]
        crop = min(h, w)
        if self.center_crop and h != w:
            image = image[(h - crop) // 2: (h + crop) // 2, (w - crop) // 2: (w + crop) // 2]
        if self.size != crop:
            interp = cv2.INTER_AREA if self.size < crop else cv2.INTER_CUBIC
            image = cv2.resize(image, dsize=(self.size, self.size), interpolation=interp)
        if random.random() < self.flip_p:
            image = cv2.flip(image, 1)
        image = cv2.GaussianBlur(image, ksize=(1, 1), sigmaX=0.2, sigmaY=0.2)
        image = ((image / 255. - 0.5) / 0.5).astype(np.float32)
        return image


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

