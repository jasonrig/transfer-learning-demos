from collections import OrderedDict

from TransferLearningDemo.utils import get_imagenet_mappings

IMAGENET_MEAN = [123.68, 116.779, 103.939]  # RGB image means
IMAGENET_MAPPINGS = get_imagenet_mappings()
SAMPLE_IMAGES = OrderedDict((
    ("dog", "https://c1.staticflickr.com/9/8083/8299709853_fc69615369_b_d.jpg"),
    ("penguin", "https://c1.staticflickr.com/5/4150/5041694815_cdab2d62f5_b_d.jpg"),
    ("eggplant", "https://c2.staticflickr.com/2/1310/5100871059_0c88627b15_b_d.jpg"),
    ("lizard", "https://c2.staticflickr.com/8/7031/6522863165_4f39a6c240_b_d.jpg"),
    ("traffic light", "https://c2.staticflickr.com/2/1599/25818250105_8fa094f6d8_b_d.jpg"),
    ("stop sign", "https://c1.staticflickr.com/1/9/13019768_54eda3afc8_o_d.jpg"),
    ("beer bottle", "https://c1.staticflickr.com/7/6186/6117191681_419bd8225f_b_d.jpg"),
    ("keyboard", "https://c1.staticflickr.com/1/65/212624192_dc07a01499_b_d.jpg"),
    ("snail", "https://c2.staticflickr.com/8/7117/7060260777_1474f01cfa_b_d.jpg"),
))
