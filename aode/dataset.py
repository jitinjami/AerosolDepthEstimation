import os
from typing import Any

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from tifffile import imread
from torch.utils.data import Dataset


class SolaDataset(Dataset):
    '''A PyTorch Dataset class for reading AOD Train Dataset from Solafune.
    Each image has 13 channels and a resolution of 128x128. This dataset
    allows optional caching of images in memory to speed up data loading and
    includes automatic mean and standard deviation calculation for
    normalization if loading data into cache.
    '''

    def __init__(
        self,
        path_csv: str,
        dir_img: str,
        test: bool,
        mean: list[float],
        std: list[float],
        cache: bool = False,
    ) -> None:
        '''A PyTorch Dataset class for reading AOD Train Dataset from Solafune.
        Each image has 13 channels and a resolution of 128x128. This dataset
        allows optional caching of images in memory to speed up data loading
        and includes automatic mean and standard deviation calculation for
        normalization if loading data into cache.

        Args:
            path_csv (str): Path to the CSV file containing image filenames
            and labels.
            dir_img (str): Path to the directory of images files.
            test (bool): Whether to set up for test phase.
            mean (list[float]): Mean values for image normalization.
            std (list[float]): Standard deviation values for image
            normalization.
            cache (bool, optional): Whether to cache images to memory.
            Defaults to False.
        '''
        super().__init__()

        self.path_csv = path_csv
        self.dir_img = dir_img
        self.test = test
        self.cache = cache
        self.transform = (
            A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Normalize(mean, std),
                    ToTensorV2(),
                ]
            )
            if test
            else A.Compose(A.Normalize(), ToTensorV2())
        )

        self.csv_data = np.genfromtxt(path_csv, delimiter=",", dtype=str)
        self.image_paths = [
            os.path.join(dir_img, image_name)
            for image_name in self.csv_data[:, 0]
        ]
        self.aod_values = torch.from_numpy(
            self.csv_data[:, -1].astype(np.float32)
        )

        # If cache, images and associated attributes are read into memory
        if self.cache:
            self.images = torch.empty(
                (len(self.image_paths), 13, 128, 128), dtype=torch.float32
            )

            for i, image_path in enumerate(self.image_paths):
                image = imread(image_path)
                self.images[i] = torch.tensor(image, dtype=torch.float32)

    def __len__(self) -> int:
        '''Get number of samples.

        Returns:
            int: Number of samples.
        '''
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        '''Get items.

        Args:
            index (int): Index of image

        Returns:
            tuple(torch.Tensor, torch.Tensor | None): Images and, labels (for
            train and val phase) or None (for test phase).
        '''
        if self.cache:
            image = self.images[index]
        elif not self.cache:
            image = torch.tensor(
                imread(self.image_paths[index]), dtype=torch.float32
            )

        if self.transform:
            image = self.transform(image)

        return image, self.aod_values[index]
