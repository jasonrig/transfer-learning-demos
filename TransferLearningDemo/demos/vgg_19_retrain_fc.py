import os
import random
import sys
import zipfile

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.contrib.slim.python.slim.nets import vgg
from tensorflow.python.estimator.warm_starting_util import WarmStartSettings

from TransferLearningDemo.demos import IMAGENET_MEAN
from TransferLearningDemo.utils import download, get_dirs, delete_file_safely, get_model_checkpoint

TF_RECORDS_FILE_TRAIN = os.path.join(get_dirs()['user_cache_dir'], 'retina_train.tfrecords')
TF_RECORDS_FILE_TEST = os.path.join(get_dirs()['user_cache_dir'], 'retina_test.tfrecords')

if not os.path.isfile(TF_RECORDS_FILE_TRAIN) or not os.path.isfile(TF_RECORDS_FILE_TEST):
    annotations = download("http://cecas.clemson.edu/~ahoover/stare/manifestations/annotations.zip")
    image_zip = download("http://cecas.clemson.edu/~ahoover/stare/images/all-images.zip")

    try:
        with zipfile.ZipFile(image_zip) as image_file_archive:
            with zipfile.ZipFile(annotations) as annotations_file_archive:
                imgs = image_file_archive.namelist()
                random.shuffle(imgs)
                train_imgs = imgs[:-75]
                test_imgs = imgs[-75:]
                for TF_RECORDS_FILE, img_files in [(TF_RECORDS_FILE_TRAIN, train_imgs),
                                                   (TF_RECORDS_FILE_TEST, test_imgs)]:
                    with tf.python_io.TFRecordWriter(TF_RECORDS_FILE) as writer:
                        for file_name in img_files:
                            annotation_txt_file = "%s.fea.mg.txt" % file_name.split("/")[-1].split('.')[0]
                            try:
                                with annotations_file_archive.open(annotation_txt_file) as annotation_txt:
                                    label = annotation_txt.read().decode('ascii')[11] == "1"
                                    if label:
                                        label = np.array([1, 0])
                                    else:
                                        label = np.array([0, 1])
                                with image_file_archive.open(file_name) as image_file:
                                    img = np.array(Image.open(image_file).resize((224, 224))).reshape((224 * 224 * 3))
                                    example = tf.train.Example(features=tf.train.Features(
                                        feature={
                                            'image': tf.train.Feature(int64_list=tf.train.Int64List(value=img)),
                                            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label))
                                        }
                                    ))
                                    writer.write(example.SerializeToString())
                            except KeyError:
                                pass
    except Exception:
        delete_file_safely(TF_RECORDS_FILE_TRAIN)
        delete_file_safely(TF_RECORDS_FILE_TEST)
        raise


def input_fn(test=False, batch_size=100, epochs=1):
    def _input_fn():
        def augment(img):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            rnd_crop_x = tf.random_uniform([], 200, 224, dtype=tf.int32)
            rnd_crop_y = tf.random_uniform([], 200, 224, dtype=tf.int32)
            img = tf.random_crop(img, (rnd_crop_x, rnd_crop_y, 3))
            img = tf.squeeze(tf.image.resize_images([img], (224, 224)))
            return img
        dataset = tf.data.TFRecordDataset(TF_RECORDS_FILE_TEST if test else TF_RECORDS_FILE_TRAIN)
        dataset = dataset.repeat(epochs)
        if not test:
            dataset = dataset.shuffle(100)
        dataset = dataset.map(lambda example: tf.parse_single_example(example, features={
            'image': tf.FixedLenFeature((224 * 224 * 3,), tf.int64),
            'label': tf.FixedLenFeature((2,), tf.int64)
        }))
        dataset = dataset.map(lambda example: (tf.reshape(example['image'], (224, 224, 3)), example['label']))
        if not test:
            dataset = dataset.map(
                lambda img, label: (augment(img), label))
        dataset = dataset.batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()

    return _input_fn


def input_fn_predict(file_names):
    def _input_fn():
        def decode_img(file_content):
            img = tf.image.decode_jpeg(file_content)
            img = tf.image.resize_images([img], (224,224))
            return img

        dataset = tf.data.Dataset.from_tensor_slices(tf.constant(file_names, tf.string))
        dataset = dataset.map(lambda file_name: tf.read_file(file_name))
        dataset = dataset.map(decode_img)
        return dataset.make_one_shot_iterator().get_next()
    return _input_fn


def model_fn(features, labels, mode):
    img_mean = tf.reshape(tf.constant(IMAGENET_MEAN), (1, 1, 3))
    output = vgg.vgg_19(tf.cast(features, tf.float32) - img_mean, is_training=(mode == tf.estimator.ModeKeys.TRAIN))
    logits = tf.layers.dense(tf.layers.flatten(output[1]['vgg_19/fc7']), 2, activation=None, name="new_logits")

    probabilities = tf.nn.softmax(logits)
    predicted_classes = tf.argmax(probabilities, axis=1)

    loss = None
    train_op = None

    if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
        loss = tf.losses.softmax_cross_entropy(labels, logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            train_vars = [
                tf.get_variable('new_logits/kernel'),
                tf.get_variable('new_logits/bias')
            ]
            train_vars.extend([var for var in tf.trainable_variables() if
                               var.name.startswith('vgg_19/fc') and not var.name.startswith != "vgg_19/fc8"])
            print(train_vars)
            train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(
                loss, global_step=tf.train.get_or_create_global_step(), var_list=train_vars)

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

model_dir = os.path.join(get_dirs()['user_cache_dir'], 'retina_model')
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
estimator = tf.estimator.Estimator(model_fn, warm_start_from=ws, model_dir=model_dir)

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    if len(sys.argv) == 1:
        print("Arguments must be one of:")
        print("train")
        print("evaluate")
        print("predict <file names>")
    elif sys.argv[1] == 'train':
        estimator.train(input_fn(test=False, batch_size=100, epochs=100))
        print(estimator.evaluate(input_fn(test=True, batch_size=100, epochs=1)))
    elif sys.argv[1] == 'evaluate':
        print(estimator.evaluate(input_fn(test=True, batch_size=100, epochs=1)))
    elif len(sys.argv) > 2 and sys.argv[1] == 'predict':
        print(list(estimator.predict(input_fn_predict(sys.argv[2:]))))
    else:
        print("Invalid options selected")