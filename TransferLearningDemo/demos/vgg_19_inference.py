import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.contrib.slim.python.slim.nets import vgg

from TransferLearningDemo.demos import IMAGENET_MEAN, IMAGENET_MAPPINGS, SAMPLE_IMAGES
from TransferLearningDemo.utils import download, get_model_checkpoint

graph = tf.Graph()
IMG_SIZE = (224, 224)

with graph.as_default():
    IMG_MEAN = tf.reshape(tf.constant(IMAGENET_MEAN), (1, 1, 3))

    def decode_and_resize_img(img_bytes):
        img_decoded = tf.image.decode_image(img_bytes)
        img_decoded.set_shape((None, None, 3))
        img_resized = tf.image.resize_images([img_decoded], IMG_SIZE)
        return tf.squeeze(img_resized)

    input_image_files = tf.placeholder(tf.string, (None,))
    input_images = tf.map_fn(lambda file_name: tf.read_file(file_name), input_image_files)
    input_decoded = tf.map_fn(decode_and_resize_img, input_images, dtype=tf.float32)
    input_normalised = input_decoded - IMG_MEAN

    output = vgg.vgg_19(input_normalised, is_training=False)
    scores = tf.nn.softmax(output[0])

    print("Trainable parameters: %i" % int(np.sum([np.prod(v.shape) for v in tf.trainable_variables()])))

    saver = tf.train.Saver()


def run_network(image_urls):
    img_files = [download(url) for url in image_urls]

    with tf.Session(graph=graph) as sess:
        saver.restore(sess, get_model_checkpoint("VGG 19"))
        result = sess.run([scores, output], feed_dict={input_image_files: img_files})
        return {'probabilities': result[0], 'output': result[1]}


def decode_result(result):
    class_ids = np.argmax(result['probabilities'], axis=1)
    probabilities = np.max(result['probabilities'], axis=1) * 100
    class_labels = [IMAGENET_MAPPINGS[id] for id in class_ids]
    return zip(class_ids, probabilities, class_labels)


def do_inference(image_urls):
    return dict(zip(image_urls, decode_result(run_network(image_urls))))


if __name__ == '__main__':

    result = do_inference(SAMPLE_IMAGES.values())

    # Set up some plotting
    fig, ax = plt.subplots(3, 3, figsize=(10, 10))
    ax = ax.flatten()

    for i, ((true_label, url), (label_id, score, predicted_label)) in enumerate(
            zip(SAMPLE_IMAGES.items(), result.values())):
        ax[i].set_title("(%0.1f%%)\n$\\bf{%s}$:\n%s" % (score, r'\/'.join(true_label.split()), predicted_label))
        ax[i].imshow(Image.open(download(url)))
    plt.show()
