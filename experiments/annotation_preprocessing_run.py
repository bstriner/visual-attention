import json

from visual_attention.annotation_preprocessing import load_file, calc_annotations


def main():
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

def main2():
    path = r'D:\Projects\data\mscoco\2017\annotations\captions_train2017.json'
    data = load_file(path)
    ann = calc_annotations(data, output_path='output/processed-annotations')


if __name__ == '__main__':
    main()
