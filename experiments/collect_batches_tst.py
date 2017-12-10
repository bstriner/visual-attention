import json
import os

import numpy as np


def merge_batches_tst(annotation_path, image_path_fmt, output_fmt, splits):
    with open(annotation_path, 'r') as f:
        annotation_data = json.load(f)
    image_ids = np.array([img['id'] for img in annotation_data['images']], np.int32)
    n = image_ids.shape[0]
    batch_size = n // splits
    for i in range(n):
        i0 = i*batch_size
        i1 = i0 + batch_size
        if i1 > n:
            i1 = n
        batch_n = i1-i0
        batch_image_ids = image_ids[i0:i1]
        batch_images = np.zeros((batch_n, 14, 14, 512), dtype=np.float32)
        for j in range(batch_n):
            image_id = batch_image_ids[j]
            image_path = image_path_fmt.format(image_id)
            image_feats = np.load(image_path)
            batch_images[j, :, :, :] = image_feats
        batch_path = output_fmt.format(i)
        np.savez(batch_path, images=batch_images, image_ids=image_ids)
    print("Image count: {}".format(n))


def main():
    splits = 10
    annotation_path = os.path.join(os.environ['MSCOCO_PATH'], 'image_info_test2017.json')
    os.makedirs('output/batches', exist_ok=True)
    merge_batches_tst(annotation_path=annotation_path,
                      image_path_fmt='output/features/test/{:012d}.npy',
                      output_fmt='output/batches/test-{}.npz',
                      splits=splits)


if __name__ == '__main__':
    main()
