import os

import keras
import pandas as pd
import numpy as np
import imageio
import cv2
from pathlib import Path
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10

MAX_SEQ_LENGTH = 4
NUM_FEATURES = 2048

DATA_DIR = '../../data'

gs_df = pd.read_csv(os.path.join(DATA_DIR, 'gs.csv'))
abs_starts = []
abs_ends = []
for _,row in gs_df.iterrows():
    seg_start = int(row.segment.split("_")[-2])
    abs_start = seg_start + row.start
    abs_end = seg_start + row.end
    abs_starts.append(abs_start)
    abs_ends.append(abs_end)
gs_df['abs_start'] = abs_starts
gs_df['abs_end'] = abs_ends

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def load_image(filename):
    im = imageio.imread(filename)
    im = crop_center_square(im)
    return cv2.resize(im, (IMG_SIZE, IMG_SIZE))


def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()
def extract_features(images):
    global counter
    frame_features = np.zeros(
        shape=(len(images), NUM_FEATURES), dtype="float32"
    )
    for i, batch in enumerate(images):
        frame_features[i, :] = feature_extractor.predict(
            batch[None, :], verbose=0,
        )

    return frame_features

for root, dirs, files in os.walk(f'{DATA_DIR}/all_faces'):
    n = len(dirs)
    c = 0
    for dir in dirs:
        c += 1
        name = dir.split('_')[0]
        p = Path(f'{DATA_DIR}/features/keras/{name}/{dir}.csv')
        if p.is_file():
            continue
        df = pd.DataFrame(columns=['timestamp','segment_id'] + list(range(NUM_FEATURES)))
        # get all files in the directory
        files = os.listdir(f'{DATA_DIR}/all_faces/{dir}')
        timestamps = np.array([int(file.split('.')[0]) for file in files])
        images = [load_image(f'{DATA_DIR}/all_faces/{dir}/{file}') for file in files]
        feats = extract_features(images)
        for i,t in enumerate(timestamps):
            df.loc[i] = [t,dir] + list(feats[i])
        path = Path(f'{DATA_DIR}/features/keras/{name}/')
        path.mkdir(parents=True, exist_ok=True)
        df.to_csv(f'{DATA_DIR}/features/keras/{name}/{dir}.csv', index=False)
        print(f"{dir}: {c} / {n}")





