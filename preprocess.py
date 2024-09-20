import argparse
import numpy as np
import csv
import json
import os

import sklearn
import sklearn.model_selection
from tifffile import imread
import yaml
from tqdm import tqdm


def main(cfgs: argparse.Namespace) -> None:
    with open(cfgs.path_csv, 'r') as f:
        labels = list(csv.reader(f))

    train, val = sklearn.model_selection.train_test_split(
        labels, test_size=cfgs.r_val, random_state=cfgs.seed
    )

    with open(os.path.join(cfgs.dir_data, 'train.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(train)

    with open(os.path.join(cfgs.dir_data, 'val.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(val)

    imgs = np.stack(
        [
            imread(os.path.join(cfgs.dir_img, sample[0]))
            for sample in tqdm(labels)
        ]
    )
    meta = {
        'mean': np.mean(imgs, (0, 1, 2)).tolist(),
        'std': np.std(imgs, (0, 1, 2)).tolist(),
    }

    with open(os.path.join(cfgs.dir_data, 'meta.json'), 'w') as f:
        json.dump(meta, f)


if __name__ == '__main__':
    path_cfgs = 'configs/preprocess.yaml'

    with open(path_cfgs, 'r') as f:
        cfgs = yaml.safe_load(f)
        cfgs = argparse.Namespace(**cfgs)

    main(cfgs)
