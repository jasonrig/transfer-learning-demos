import math

import matplotlib.pyplot as plt
import numpy as np
import sys

from TransferLearningDemo.demos import SAMPLE_IMAGES
from TransferLearningDemo.demos.vgg_19_inference import run_network


def visualise_filters(conv_layer=(1, 1), img=SAMPLE_IMAGES['snail']):
    assert len(conv_layer) == 2 and all([isinstance(item, int) for item in
                                         conv_layer]), "`conv_filter` must be a tuple of two integers that identify the conv layer in the VGG 19 network"
    layer_name = 'vgg_19/conv{0}/conv{0}_{1}'.format(*conv_layer)
    result = run_network([img])
    activation_maps = result['output'][1][layer_name][0]
    activation_maps = np.transpose(activation_maps, (2, 0, 1))

    n_subplots = math.ceil(math.sqrt(activation_maps.shape[0]))
    fig, ax = plt.subplots(n_subplots, n_subplots, figsize=(10, 10))
    ax = ax.flatten()

    for i, activation in enumerate(activation_maps):
        subplt = ax[i]
        subplt.imshow(activation)
        subplt.axis('off')
    plt.show()


if __name__ == "__main__":
    conv_layer = (1, 1)
    if len(sys.argv) == 3:
        conv_layer = (int(sys.argv[1]), int(sys.argv[2]))
    visualise_filters(conv_layer)
