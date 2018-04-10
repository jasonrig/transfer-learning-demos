import os
import random
import re
import zipfile

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.contrib.slim.python.slim.nets import vgg
from tensorflow.python.estimator.warm_starting_util import WarmStartSettings

from TransferLearningDemo.utils import download, get_dirs, delete_file_safely, get_model_checkpoint

TF_RECORDS_FILE_TRAIN = os.path.join(get_dirs()['user_cache_dir'], 'smiles_train.tfrecords')
TF_RECORDS_FILE_TEST = os.path.join(get_dirs()['user_cache_dir'], 'smiles_test.tfrecords')

if not os.path.isfile(TF_RECORDS_FILE_TRAIN) or not os.path.isfile(TF_RECORDS_FILE_TEST):
    non_smiles = download(
        "https://data.mendeley.com/datasets/yz4v8tb3tp/5/files/7274bb3d-ab04-4656-8188-3d2218d248ca/NON-SMILE_list.txt?dl=1",
        target_file_name="non_smile_labels.txt")
    smiles = download(
        "https://data.mendeley.com/datasets/yz4v8tb3tp/5/files/72b9b8f7-72be-4559-b730-29c41faad4d9/SMILE_list.txt?dl=1",
        target_file_name="smile_labels.txt")
    image_zip = download("http://conradsanderson.id.au/lfwcrop/lfwcrop_color.zip")

    all_data = []
    with open(non_smiles, 'r') as f:
        non_smiles = [re.sub(r'\.jpg', '.ppm', line.strip()) for line in f.readlines() if len(line.strip()) > 0]
        all_data.extend([(file_name, np.array([1, 0])) for file_name in non_smiles])

    with open(smiles, 'r') as f:
        smiles = [re.sub(r'\.jpg', '.ppm', line.strip()) for line in f.readlines() if len(line.strip()) > 0]
        all_data.extend([(file_name, np.array([0, 1])) for file_name in non_smiles])

    random.shuffle(all_data)

    train_data = all_data[:-100]
    test_data = all_data[-100:]

    try:
        for TF_RECORDS_FILE, data in [(TF_RECORDS_FILE_TRAIN, train_data), (TF_RECORDS_FILE_TEST, test_data)]:
            with tf.python_io.TFRecordWriter(TF_RECORDS_FILE) as writer:
                with zipfile.ZipFile(image_zip, 'r') as image_file_archive:
                    for file_name, label in data:
                        with image_file_archive.open("lfwcrop_color/faces/%s" % file_name) as image_file:
                            img = np.array(Image.open(image_file)).reshape((64 * 64 * 3))
                            example = tf.train.Example(features=tf.train.Features(
                                feature={
                                    'image': tf.train.Feature(int64_list=tf.train.Int64List(value=img)),
                                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label))
                                }
                            ))
                            writer.write(example.SerializeToString())
    except Exception:
        delete_file_safely(TF_RECORDS_FILE_TRAIN)
        delete_file_safely(TF_RECORDS_FILE_TEST)
        raise


def input_fn(test=False, batch_size=100):
    def _input_fn():
        dataset = tf.data.TFRecordDataset(TF_RECORDS_FILE_TEST if test else TF_RECORDS_FILE_TRAIN)
        if not test:
            dataset = dataset.shuffle(100)
        dataset = dataset.map(lambda example: tf.parse_single_example(example, features={
            'image': tf.FixedLenFeature((64 * 64 * 3,), tf.int64),
            'label': tf.FixedLenFeature((2,), tf.int64)
        }))
        dataset = dataset.map(lambda example: (tf.reshape(example['image'], (64, 64, 3)), example['label']))
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(lambda images, labels: (tf.image.resize_images(images, (224, 224)), labels))
        return dataset.make_one_shot_iterator().get_next()

    return _input_fn


def model_fn(features, labels, mode):
    output = vgg.vgg_19(features, is_training=(mode == tf.estimator.ModeKeys.TRAIN))
    logits = tf.layers.dense(tf.layers.flatten(output[1]['vgg_19/fc7']), 2, activation=None, name="new_logits")
    loss = tf.losses.softmax_cross_entropy(labels, logits)
    probabilities = tf.nn.softmax(logits)
    predicted_classes = tf.argmax(probabilities, axis=1)

    train_op = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            train_op = tf.train.AdamOptimizer(learning_rate=0.005).minimize(
                loss, global_step=tf.train.get_or_create_global_step(), var_list=[
                    tf.get_variable('new_logits/kernel'),
                    tf.get_variable('new_logits/bias')
                ])

    metrics = None
    if mode == tf.estimator.ModeKeys.EVAL:
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=tf.argmax(labels, axis=1), predictions=predicted_classes)
        }

    predictions = None
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': probabilities,
            'predictions': predicted_classes
        }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics)


ws = WarmStartSettings(
    ckpt_to_initialize_from=get_model_checkpoint('VGG 19'),
    vars_to_warm_start="vgg_19/.*"
)

model_dir = os.path.join(get_dirs()['user_cache_dir'], 'face_model')
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
estimator = tf.estimator.Estimator(model_fn, warm_start_from=ws, model_dir=model_dir)

if __name__ == "__main__":
    estimator.train(input_fn(batch_size=10))
