import csv
import json
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


def collect_vocab(annotations, mincount=10):
    counts = {}
    for a in annotations:
        caption = a['caption']
        tokens = tokenize(caption)
        for t in tokens:
            if t in counts:
                counts[t] += 1
            else:
                counts[t] = 1
    vocab = list(counts.keys())
    vocab = [v for v in vocab if counts[v] >= mincount]
    vocab.sort()
    counts = {k: v for k, v in counts.items() if k in vocab}
    return vocab, counts


def write_vocab(path, vocab, counts):
    with open(path + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Id', 'Word', 'Counts'])
        for i, v in enumerate(vocab):
            w.writerow([i, v, counts[v]])
    np.save(path + '.npy', np.array(vocab, dtype=np.string_))


def token_id(vmap, token):
    if token in vmap:
        return vmap[token] + 1
    else:
        return 0


def annotation_vector(vmap, caption):
    tokens = tokenize(caption)
    vec = np.array([token_id(vmap, t) + 2 for t in tokens] + [1], dtype=np.int32)
    return vec


def combine_vectors(vs):
    n = len(vs)
    m = max(v.shape[0] for v in vs)
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


def calc_vocab(data, mincount=10):
    annotations = data['annotations']
    vocab, counts = collect_vocab(annotations, mincount)
    vmap = {k: i for i, k in enumerate(vocab)}
    return vocab, vmap, counts


def calc_annotations(data, vmap):
    annotations = data['annotations']
    vecs = annotation_vectors(vmap, annotations)
    keys = list(vecs.keys())
    keys.sort()
    vecs = [vecs[k] for k in keys]
    return np.array(vecs), np.array(keys)


def write_annotation(data, vmap, output_path):
    ann, img = calc_annotations(data, vmap=vmap)
    kw = {'annotations': ann,
          'image_ids': img}
    np.savez(output_path, **kw)
