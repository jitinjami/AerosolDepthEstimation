import os
from typing import Any

import albumentations as A
import numpy as np
import rasterio
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class SolaTrainDataset(Dataset):
    '''
    A PyTorch Dataset class for reading AOD Train Dataset from Solafune. Each image has 13 channels and a resolution of 128x128. This dataset allows optional caching
    of images in memory to speed up data loading and includes automatic mean and standard deviation
    calculation for normalization if loading data into cache.

    Attributes:
    -----------
    path_csv : str
        Path to the CSV file containing image filenames and additional information.

    dir_img : str
        Directory where the image files are located.

    cache : bool, optional
        If True, all images are loaded into memory during initialization (default is False).

    transform : callable, optional
        Optional transform to be applied to the images (default is None).
    '''

    def __init__(
        self,
        path_csv: str,
        dir_img: str,
        phase: str,
        cache: bool = False,
    ) -> None:
        super().__init__()

        self.path_csv = path_csv
        self.dir_img = dir_img
        self.cache = cache
        self.transform = (
            A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )
            if phase == 'train'
            else A.Compose(A.Normalize(), ToTensorV2())
        )

        # Read CSV using Numpy
        self.csv_data = np.genfromtxt(path_csv, delimiter=",", dtype=str)

        # Get image paths as a list
        self.image_paths = [
            os.path.join(dir_img, image_name)
            for image_name in self.csv_data[:, 0]
        ]

        # Get aod values and convert to tensor
        self.aod_values = torch.from_numpy(
            self.csv_data[:, -1].astype(np.float32)
        )

        # If cache is activated, images and associated attributes are read into memory
        if self.cache:
            self.images = torch.empty(
                (len(self.image_paths), 13, 128, 128), dtype=torch.float32
            )

            for i, image_path in enumerate(self.image_paths):
                image = np.array(rasterio.open(image_path).read())
                self.images[i] = torch.tensor(image, dtype=torch.float32)

            self.means = self.images.mean(dim=(0, 2, 3))
            self.stds = self.images.std(dim=(0, 2, 3))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Any:
        '''Get items.

        Args:
            index (int): Index of image

        Returns:
            tuple(torch.Tensor, torch.float)
        '''
        # TODO: data augmentation
        aod_value = self.aod_values[index]

        if self.cache:
            image = self.images[index]

        elif not self.cache:
            image = torch.tensor(
                np.array(rasterio.open(self.image_paths[index]).read()),
                dtype=torch.float32,
            )

        if self.transform:
            image = self.transform(image)

        return (image, aod_value)


class SolaTestDataset(Dataset):
    '''
    A PyTorch Dataset class for reading AOD Test Dataset from Solafune. Each image has 13 channels and a resolution of 128x128. This dataset allows optional caching
    of images in memory to speed up data loading and includes automatic mean and standard deviation
    calculation for normalization if loading data into cache.

    Attributes:
    -----------
    path_csv : str
        Path to the CSV file containing image filenames and additional information.

    dir_img : str
        Directory where the image files are located.

    cache : bool, optional
        If True, all images are loaded into memory during initialization (default is False).

    transform : callable, optional
        Optional transform to be applied to the images (default is None).
    '''

    def __init__(
        self, path_csv: str, dir_img: str, cache: bool = False
    ) -> None:
        super().__init__()

        self.path_csv = path_csv
        self.dir_img = dir_img
        self.cache = cache

        # Read CSV using Numpy
        self.csv_data = np.genfromtxt(path_csv, delimiter=",", dtype=str)

        # Get image paths as a list
        self.image_paths = [
            os.path.join(dir_img, image_name)
            for image_name in self.csv_data[:, 0]
        ]

        # If Cache is activated, images and associated attributes are read into memory
        if self.cache:
            self.images = torch.empty(
                (len(self.image_paths), 13, 128, 128), dtype=torch.float32
            )

            for i, image_path in enumerate(self.image_paths):
                image = np.array(rasterio.open(image_path).read())
                self.images[i] = torch.tensor(image, dtype=torch.float32)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Any:
        '''Get items.

        Args:
            index (int): Index of image

        Returns:
            torch.Tensor
        '''

        if self.cache:
            image = self.images[index]

        elif not self.cache:
            image = torch.tensor(
                np.array(rasterio.open(self.image_paths[index]).read()),
                dtype=torch.float32,
            )

        if self.transform:
            image = self.transform(image)

        return image
