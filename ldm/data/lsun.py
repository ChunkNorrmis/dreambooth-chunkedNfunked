import os, random, torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.io import decode_image


class LSUNBase(Dataset):
    def __init__(self,
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
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }
        self.chance = flip_p
        self.size = size

        mean = 0.
        std = 0.
        for data in self.image_paths:
            data = decode_image(data, mode='RGB')
            crop = min(data.shape[1], data.shape[2])
            data = fun.center_crop(data, size=(crop, crop))
            data = fun.resize(data, size=(self.size, self.size), interpolation=3, antialias=True)
            data = fun.to_dtype(data, dtype=torch.float32, scale=True)
            data = data.view(data.size(0), -1)
            mean += data.mean(1)
            std += data.std(1)
        self.mean = mean / self.num_images
        self.std = std / self.num_images

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = decode_image(example["file_path_"], mode="RGB")
        transform = v2.Compose([
            v2.ToDtype(dtype=torch.uint8, scale=True),
            v2.CenterCrop(min(image.shape[1], image.shape[2])),
            v2.Resize((self.size, self.size), interpolation=3, antialias=True),
            v2.RandomHorizontalFlip(p=self.chance),
            v2.GaussianBlur(kernel_size=1, sigma=(0.1, 0.3)),
            v2.ToDtype(dtype=torch.float32, scale=True),
            v2.Normalize(mean=self.mean, std=self.std),
            v2.Lambda(lambda tt: np.array(tt.clone().detach().permute(1, 2, 0).cpu()).astype(np.float32))
        ])
        example['image'] = transform(image)


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
