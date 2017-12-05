import os

from visual_attention.preprocessing import preprocess_dir


def run(path, cropped, features):
    sets = ['val', 'train', 'test']
    for s in sets:
        preprocess_dir(os.path.join(path, s),
                       os.path.join(cropped, s),
                       os.path.join(features, s))


def main():
    input_dir = os.environ['MSCOCO_PATH']
    preprocess_dir(input_dir, 'output/cropped', 'output/features')


if __name__ == '__main__':
    main()
