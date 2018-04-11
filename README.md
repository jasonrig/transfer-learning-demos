# Transfer Learning Demo code
This package is companion code for my presentation on convolutional
neural networks, downloadable here: [https://doi.org/10.4225/03/5acdaeadb87d0](https://doi.org/10.4225/03/5acdaeadb87d0)

The package can be installed on your system using pip:

```
pip install -U git+https://github.com/jasonrig/transfer-learning-demos
```

## Visualising network activations
```
python3 -m TransferLearningDemo.demos.vgg_19_activate_filters <conv_block> <conv_layer> <filter_index>
```

To visualise the second filter of the first convolutional layer of the fifth block, run:

```
python3 -m TransferLearningDemo.demos.vgg_19_activate_filters 5 1 1
```

Remember that filter numbers are indexed from zero, whereas the convolutional layers are indexed from one.

## Training a retinal haemorrhage detection model
Train using:
```
python3 -m TransferLearningDemo.demos.vgg_19_retrain_fc train
```

Evaluate using:
```
python3 -m TransferLearningDemo.demos.vgg_19_retrain_fc evaluate
```

Predict image(s) using:
```
python3 -m TransferLearningDemo.demos.vgg_19_retrain_fc predict <image_name>
```
Add multiple files for more than one prediction.

## Neural style transfer
To run using the default style and content images:
```
python3 -m TransferLearningDemo.demos.vgg_19_style_transfer
```

To run using your own (jpeg only):
```
python3 -m TransferLearningDemo.demos.vgg_19_style_transfer <style_image> <content_image>
```