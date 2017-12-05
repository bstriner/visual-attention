import csv
import json
import os
import re
import numpy as np

from nltk.tokenize import WordPunctTokenizer


def load_file(path):
    with open(path) as fp:
        return json.load(fp)


def tokenize(caption):
    caption = caption.lower()
    tokens = WordPunctTokenizer().tokenize(caption)
    tokens = [re.sub(r'\W+', '', t) for t in tokens]
    tokens = [t for t in tokens if t]
    return tokens
    # caption = re.sub(r'\W+', '', caption)
    # caption = re.sub(r'\s+', ' ', caption)
    # tokens = re.split(r'\s+', caption)
    # return tokens


def collect_vocab(annotations):
    vocab = set()
    for a in annotations:
        caption = a['caption']
        tokens = tokenize(caption)
        for t in tokens:
            vocab.add(t)
    vocab = list(vocab)
    vocab.sort()
    return vocab


def write_vocab(path, vocab):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Id', 'Word'])
        for i, v in enumerate(vocab):
            w.writerow([i, v])

def annotation_vector(vmap, caption):
    tokens = tokenize(caption)
    vec = np.array([vmap[t]+2 for t in tokens]+[1], dtype=np.int32)
    return vec

def combine_vectors(vs):
    n = len(vs)
    m = max(v.size[0] for v in vs)
    d = np.zeros((n, m), dtype=np.int32)
    for i, v in enumerate(vs):
        d[i, :v.shape[0]] = v
    return d

def annotation_vectors(vmap, annotations):
    vecs = {}
    for a in annotations:
        caption = a['caption']
        vec = annotation_vector(vmap, caption)
        image_id = a['image_id']
        if image_id not in vecs:
            vecs[image_id] = []
        vecs[image_id].append(vec)

    cvecs = {i: combine_vectors(vs) for i, vs in vecs.items()}
    return cvecs


def calc_annotations(data, output_path):
    os.makedirs(output_path, exist_ok=True)
    annotations = data['annotations']
    vocab = collect_vocab(annotations)
    vocab_path = os.path.join(output_path, 'vocab.txt')
    write_vocab(vocab_path, vocab)

    vmap = {k:i for i, k in enumerate(vocab)}


    """
    for a in annotations:
        caption = a['caption']
        print(caption)
        tokens = tokenize(caption)
        print(tokens)
        raise ValueError()
    """
