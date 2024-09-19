from typing import Any
from torch.utils.data import Dataset


class SolaDataset(Dataset):
    def __inti__(self, path_csv: str, dir_img: str, cache: bool = False) -> None:
        super().__init__()

        # TODO: cache images to memory

    def __getitem__(self, index) -> Any:
        '''Get items.

        Args:
            index (_type_): _description_

        Returns:
            tuple(torch.Tensor, torch.float)
        '''        
        # TODO: data augmentation

