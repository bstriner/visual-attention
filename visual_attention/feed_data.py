import numpy as np
import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunHook, SessionRunArgs


def input_fn():
    pimgs = tf.placeholder(dtype=tf.float32, shape=[None, 14, 14, 512], name='images')
    passignments = tf.placeholder(dtype=tf.int32, shape=[None], name='assignments')
    pcaptions = tf.placeholder(dtype=tf.int32, shape=[None, None], name='captions')

    kw = {'images': pimgs, 'assignments': passignments, 'captions': pcaptions}
    return kw, None


class FeedFnHook(SessionRunHook):
    def __init__(self, path_fmt, splits, batch_size):
        self.path_fmt = path_fmt
        self.splits = splits
        self.batch_size = batch_size
        self.batch_iter = self.gen_splits_forever()

    def load_placeholders(self, graph):
        placeholder_images = graph.get_tensor_by_name("images:0")
        placeholder_assignments = graph.get_tensor_by_name("assignments:0")
        placeholder_captions = graph.get_tensor_by_name("captions:0")
        return placeholder_images, placeholder_assignments, placeholder_captions

    def gen_split_batches(self, split):
        path = self.path_fmt.format(split)
        data = np.load(path)
        annotations = data['annotations']
        images = data['images']
        n = images.shape[0]
        idx = np.arange(n)
        np.random.shuffle(idx)
        batch_count = -((-n) // self.batch_size)
        for i in range(batch_count):
            i0 = i * self.batch_size
            i1 = i0 + self.batch_size
            if i1 > n:
                i1 = n
            ids = idx[i0:i1]

            batch_images = images[ids, :, :, :]
            batch_assignments = []
            cap_n = 0
            cap_d = 0
            for j, id in enumerate(ids):
                a = annotations[id]
                batch_assignments.append(np.ones((a.shape[0],), dtype=np.int32) * j)
                cap_n += a.shape[0]
                if cap_d < a.shape[1]:
                    cap_d = a.shape[1]
            batch_assignments = np.concatenate(batch_assignments, axis=0)

            batch_annotations = np.zeros((cap_n, cap_d), dtype=np.int32)
            pos = 0
            for id in ids:
                a = annotations[id]
                batch_annotations[pos:pos + a.shape[0], :a.shape[1]] = a
                pos += a.shape[0]

            yield batch_images, batch_assignments, batch_annotations

    def gen_splits(self):
        idx = np.arange(self.splits)
        np.random.shuffle(idx)
        for i in idx:
            for j in self.gen_split_batches(i):
                yield j

    def gen_splits_forever(self):
        while True:
            for i in self.gen_splits():
                yield i

    def build_feed_dict(self, graph):
        placeholder_images, placeholder_assignments, placeholder_captions = self.load_placeholders(graph)
        batch_images, batch_assignments, batch_captions = next(self.batch_iter)
        feed_dict = {
            placeholder_images: batch_images,
            placeholder_assignments: batch_assignments,
            placeholder_captions: batch_captions,
        }
        #print("batch_images: {}".format(batch_images.shape))
        #print("batch_assignments: {}".format(batch_assignments.shape))
        #print("batch_captions: {}".format(batch_captions.shape))
        #assert np.all(np.isfinite(batch_images))
        #assert np.all(np.isfinite(batch_assignments))
        #assert np.all(np.isfinite(batch_captions))
        return feed_dict

    def before_run(self, run_context):
        feed_dict = self.build_feed_dict(graph=run_context.session.graph)
        return SessionRunArgs(fetches=None, feed_dict=feed_dict)
