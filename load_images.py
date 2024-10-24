import os

import numpy as np
from PIL.Image import Image

from dataset import DATA_DIR, load_csv


def load(humor: bool):
    gs_df = load_csv(os.path.join(DATA_DIR, 'gs.csv'))
    images_train = []
    images_dev = []

    # walk directory day3/all_faces
    for root, dirs, files in os.walk(f'{DATA_DIR}/day3/all_faces'):
        for dir in dirs:
            # get all files in the directory
            files = os.listdir(f'{DATA_DIR}/day3/all_faces/{dir}')
            # get sub view of annotations
            sub_df = gs_df[gs_df.segment == dir]
            sub_df = sub_df[(sub_df.humor == int(humor))]
            starts = sub_df.abs_start.values
            ends = sub_df.abs_end.values
            splits = sub_df.partition.values
            timestamps = np.array([int(file.split('.')[0]) for file in files])
            starts = starts[None, :]
            ends = ends[None, :]
            timestamps = timestamps[:, None]
            # check if the timestamps are in the range of the starts and ends
            bounds_check = (timestamps >= starts) & (timestamps <= ends)
            is_humorous = bounds_check.any(axis=1)
            _, associated_segment = np.nonzero(bounds_check)
            index = np.nonzero(is_humorous)[0]
            files = [files[idx] for idx in index]
            for i, file in enumerate(files):
                split = splits[associated_segment[i]]
                match split:
                    case "train":
                        images_train.append(f'{DATA_DIR}/day3/all_faces/{dir}/{file}')
                    case "devel":
                        images_dev.append(f'{DATA_DIR}/day3/all_faces/{dir}/{file}')

    images_train = [Image.open(img) for img in images_train]
    images_dev = [Image.open(img) for img in images_dev]

    return images_train, images_dev