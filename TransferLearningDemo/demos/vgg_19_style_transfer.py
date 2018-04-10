import sys

import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import animation

from TransferLearningDemo.demos import IMAGENET_MEAN
from TransferLearningDemo.utils import get_model_checkpoint, download
from TransferLearningDemo.vgg import vgg_19_style_transfer

IMG_SIZE = (224, 224)


def generate_graph(graph=None):
    if graph is None:
        graph = tf.Graph()

    with graph.as_default():
        IMG_MEAN = tf.reshape(tf.constant(IMAGENET_MEAN), (1, 1, 3))

        content_image_file = tf.placeholder(shape=[], dtype=tf.string)
        style_image_file = tf.placeholder(shape=[], dtype=tf.string)

        content_image = tf.image.decode_jpeg(tf.read_file(content_image_file))
        content_image_resized = tf.image.resize_images([content_image], IMG_SIZE)
        content_image_resized.set_shape((1, IMG_SIZE[0], IMG_SIZE[1], 3))

        style_image = tf.image.decode_jpeg(tf.read_file(style_image_file))
        style_image_resized = tf.image.resize_images([style_image], IMG_SIZE)
        style_image_resized.set_shape((1, IMG_SIZE[0], IMG_SIZE[1], 3))

        generated_img_params = tf.get_variable('img_params',
                                               initializer=tf.random_normal((1, IMG_SIZE[0], IMG_SIZE[1], 3), 0, 0.01),
                                               dtype=tf.float32)
        generated_img = tf.nn.tanh(generated_img_params) * 255.0
        output_img = tf.clip_by_value(generated_img + IMG_MEAN, 0, 255)
        output_img = tf.squeeze(tf.cast(output_img, tf.uint8))


        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            vgg_output_img = vgg_19_style_transfer(generated_img, is_training=False)
            vgg_style = vgg_19_style_transfer(style_image_resized - IMG_MEAN, is_training=False)
            vgg_content = vgg_19_style_transfer(content_image_resized - IMG_MEAN, is_training=False)

        def content_diff(conv_layer):
            layer_name = 'vgg_19/conv{0}/conv{0}_{1}'.format(*conv_layer)
            l1 = vgg_content[1][layer_name]
            shp = l1.shape.as_list()
            l1 = tf.reshape(l1, (shp[1] * shp[2], shp[3]))
            l2 = vgg_output_img[1][layer_name]
            l2 = tf.reshape(l2, (shp[1] * shp[2], shp[3]))
            shp = l1.shape.as_list()
            norm_factor = (1 / (4 * shp[0] * shp[1]))
            difference = l1 - l2
            return norm_factor * tf.linalg.norm(difference)

        def style_diff(conv_layer):
            layer_name = 'vgg_19/conv{0}/conv{0}_{1}'.format(*conv_layer)
            l1 = vgg_style[1][layer_name]
            shp = l1.shape.as_list()
            l1 = tf.reshape(l1, (shp[1] * shp[2], shp[3]))
            l2 = vgg_output_img[1][layer_name]
            l2 = tf.reshape(l2, (shp[1] * shp[2], shp[3]))
            shp = l1.shape.as_list()
            norm_factor = (1 / (4 * shp[0] ** 2 * shp[1] ** 2))
            g1 = tf.matmul(tf.transpose(l1), l1)
            g2 = tf.matmul(tf.transpose(l2), l2)
            difference = g1 - g2
            return norm_factor * tf.linalg.norm(difference)

        content_loss = sum([content_diff((x, 2)) for x in [4, 5]])
        style_loss = sum([style_diff((x, 1)) for x in range(1, 6)])

        weight = tf.placeholder(shape=[], dtype=tf.float32)

        loss = (weight * style_loss) + ((1 - weight) * content_loss)

        train_op = tf.train.AdamOptimizer(0.01).minimize(loss, var_list=[generated_img_params])
        init_op = tf.global_variables_initializer()

        saver = tf.train.Saver(list(filter(lambda v: v != generated_img_params, tf.trainable_variables())))

        def init_fn(sess):
            sess.run(init_op)
            saver.restore(sess, get_model_checkpoint("VGG 19"))

        return {
            'graph': graph,
            'weight': weight,
            'style_img_file': style_image_file,
            'content_img_file': content_image_file,
            'output_img': output_img,
            'init_fn': init_fn,
            'train_op': train_op,
            'loss': loss
        }


def train(epochs, style_img, content_img, weight=0.9, yield_every=1):
    model = generate_graph()
    with tf.Session(graph=model['graph']) as sess:
        model['init_fn'](sess)
        for i in range(epochs):
            loss, _ = sess.run([model['loss'], model['train_op']], feed_dict={
                model['content_img_file']: content_img,
                model['style_img_file']: style_img,
                model['weight']: weight
            })
            print(loss)
            if i % yield_every == 0:
                yield sess.run(model['output_img']).squeeze()
        yield sess.run(model['output_img']).squeeze()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        style_img = download("https://github.com/jasonrig/transfer-learning-demos/raw/master/style.jpg")
        content_img = download("https://github.com/jasonrig/transfer-learning-demos/raw/master/content.jpg")
    else:
        style_img = sys.argv[1]
        content_img = sys.argv[2]
    fig = plt.figure(figsize=(5, 5))
    img_generator = train(200, style_img, content_img)
    im = plt.imshow(next(img_generator), animated=True)
    current_frame = 1
    plt.title(current_frame)


    def update_animation(*args):
        global current_frame
        try:
            im.set_array(next(img_generator))
            plt.title(current_frame)
            current_frame += 1
        except StopIteration:
            pass
        return im,


    ani = animation.FuncAnimation(fig, update_animation, interval=1, blit=True)

    plt.show()
