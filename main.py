from aode.models.swin import SwinTransformerV2
from aode.dataset import SolaTrainDataset, SolaTestDataset
import argparse
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

swin = SwinTransformerV2()

def main():
    #path_config = 'configs/train.yaml'

    # Data paths
    train_csv = './data/train_answer.csv'
    train_img_dir = './data/train_images'
    test_img_dir = 'test_images'

    # Transforms (Normalization for 12 channels)
    train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(),
        ToTensorV2()
    ])

    # Datasets and DataLoaders
    train_dataset = SolaTrainDataset(train_csv, train_img_dir, cache=True, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    print(train_dataset.means)

if __name__ == '__main__':
    main()
