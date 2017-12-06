import os

import numpy as np
from tqdm import tqdm


def merge_batches(annotation_path, image_path_fmt, output_fmt, splits):
    annotation_data = np.load(annotation_path)
    annotations = annotation_data['annotations']
    image_ids = annotation_data['image_ids']
    n = image_ids.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)

    batch_size = n // splits
    for i in tqdm(range(splits), desc='Merging batches'):
        i0 = i * batch_size
        i1 = i0 + batch_size
        if i1 > n:
            i1 = n
        batch_n = i1 - i0
        batch_idx = idx[i0:i1]
        batch_image_ids = image_ids[batch_idx]

        batch_images = np.zeros((batch_n, 14, 14, 512), dtype=np.float32)
        for j in range(batch_n):
            image_id = batch_image_ids[j]
            image_path = image_path_fmt.format(image_id)
            image_feats = np.load(image_path)
            batch_images[j, :, :, :] = image_feats

        batch_annotations = annotations[batch_idx]
        batch_path = output_fmt.format(i)
        np.savez(batch_path, annotations=batch_annotations, images=batch_images)

    print("Image count: {}".format(n))


def main():
    splits = 20
    os.makedirs('output/batches', exist_ok=True)
    merge_batches('output/processed-annotations/train-annotations.npy',
                  'output/features/train/{:011d}.jpg',
                  'output/batches/train-{}.npz',
                  splits=splits)
    merge_batches('output/processed-annotations/val-annotations.npy',
                  'output/features/val/{:011d}.jpg',
                  'output/batches/val.npz',
                  splits=1)
    pass


if __name__ == '__main__':
    main()
