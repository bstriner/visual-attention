import json
import os

from visual_attention.annotation_preprocessing import load_file, write_annotation, calc_vocab, write_vocab


def main2():
    path = r'D:\Projects\data\mscoco\2017\annotations\captions_val2017.json'
    with open(path) as fp:
        x = json.load(fp)
    print(x.keys())
    print(x['info'])
    print(x['images'][0])
    print(x['annotations'][0])
    path = r'D:\Projects\data\mscoco\2017\annotations\image_info_test2017.json'
    x = load_file(path)
    print(x.keys())
    print(x['info'])
    print(x['images'][0])
    print(len(x['images']))
    print(x['categories'][0])
    print(len(x['categories']))
    path = r'D:\Projects\data\mscoco\2017\annotations\image_info_test-dev2017.json'
    x = load_file(path)
    print(x.keys())
    print(x['categories'][0])


def main():
    basepath = r'D:\Projects\data\mscoco\2017\annotations'
    output_path = 'output/processed-annotations'
    os.makedirs(output_path, exist_ok=True)
    mincount = 10

    data = load_file(os.path.join(basepath, 'captions_train2017.json'))
    val = load_file(os.path.join(basepath, 'captions_val2017.json'))

    vocab, vmap, counts = calc_vocab(data, mincount=mincount)
    write_vocab(os.path.join(output_path, 'vocab'), vocab=vocab, counts=counts)
    print('Vocabulary size: {}'.format(len(vocab)))
    write_annotation(data=data, vmap=vmap, output_path=os.path.join(output_path, 'train-annotations.npz'))
    write_annotation(data=val, vmap=vmap, output_path=os.path.join(output_path, 'val-annotations.npz'))


if __name__ == '__main__':
    main()
