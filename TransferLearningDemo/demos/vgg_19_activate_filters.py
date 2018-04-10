import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib import animation
from tensorflow.contrib.slim.python.slim.nets import vgg

from TransferLearningDemo.demos import IMAGENET_MEAN
from TransferLearningDemo.utils import get_model_checkpoint


def generate_graph(conv_layer=(1, 1), filter_idx=0, lr=0.01, graph=None):
    assert len(conv_layer) == 2 and all([isinstance(item, int) for item in
                                         conv_layer]), "`conv_filter` must be a tuple of two integers that identify the conv layer in the VGG 19 network"
    layer_name = 'vgg_19/conv{0}/conv{0}_{1}'.format(*conv_layer)

    if graph is None:
        graph = tf.get_default_graph()

    with graph.as_default():
        img_mean = tf.reshape(tf.constant(IMAGENET_MEAN), (1, 1, 3))
        params = tf.get_variable("synthetic_img",
                                 initializer=tf.random_normal((1, 224, 224, 3), 0, 0.001, dtype=tf.float32))
        img = (tf.nn.tanh(params) * 255.0) + img_mean
        img = tf.clip_by_value(img, 0, 255)
        test_img = tf.cast(img, tf.uint8)

        output = vgg.vgg_19(img - img_mean, is_training=False)

        layer = output[1][layer_name]
        loss = -tf.reduce_mean(layer[:, :, :, filter_idx])
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=[params])

        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver(list(filter(lambda v: v != params, tf.trainable_variables())))

        def init_fn(sess):
            sess.run(init_op)
            saver.restore(sess, get_model_checkpoint("VGG 19"))

        return {
            'graph': graph,
            'test_img': test_img,
            'init_fn': init_fn,
            'train_op': train_op,
            'loss': loss
        }


def train(epochs, conv_layer, filter_idx, yield_every=1):
    model = generate_graph(conv_layer, filter_idx)
    with tf.Session(graph=model['graph']) as sess:
        model['init_fn'](sess)
        for i in range(epochs):
            sess.run(model['train_op'])
            if i % yield_every == 0:
                yield sess.run(model['test_img']).squeeze()
        yield sess.run(model['test_img']).squeeze()


if __name__ == "__main__":
    conv_layer = (5, 1)
    filter_idx = 4
    if len(sys.argv) == 4:
        conv_layer = (int(sys.argv[1]), int(sys.argv[2]))
        filter_idx = int(sys.argv[3])

    fig = plt.figure(figsize=(5, 5))
    img_generator = train(50, conv_layer, filter_idx)
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