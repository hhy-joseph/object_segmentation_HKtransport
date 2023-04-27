# MMSeg Model with 3 Classes: Bus, Train, and Taxi(dc9)

Classes: {背景:bg,巴膠:bus,dc9:taxi,電車:train}

This is a readme file for an MMSeg model trained on a dataset with 3 classes: bus, train, and taxi(dc9).

## Model Architecture
The model architecture is an Encoder-Decoder with ResNet50_v1c as the backbone. The backbone has 4 stages, with dilations of (1, 1, 2, 4) and strides of (1, 2, 1, 1). The model uses batch normalization, with type='BN' and requires_grad=True.

The model's decode_head is a PSPHead with an input channel of 2048 and an output channel of 512. It uses pyramid pooling with pool scales of (1, 2, 3, 6) and has a dropout ratio of 0.1. The decode_head outputs 4 classes and uses CrossEntropyLoss with a loss weight of 1.0.

The auxiliary_head is an FCNHead with an input channel of 1024 and an output channel of 256. It uses a single convolutional layer and has a dropout ratio of 0.1. The auxiliary_head outputs 4 classes and uses CrossEntropyLoss with a loss weight of 0.4.

## Dataset
The dataset used for training is the Stanford Background Dataset. The data is stored in /content/drive/MyDrive/transport_segmentation and consists of images and corresponding ground-truth semantic segmentation maps. The images are resized to (320, 240) and randomly cropped to (256, 256) during training. The images are also augmented with random flips and photometric distortion.

## Training
The model is trained using Stochastic Gradient Descent (SGD) with a learning rate of 0.01, momentum of 0.9, and weight decay of 0.0005. The learning rate is decreased with a polynomial decay policy with a power of 0.9 and a minimum learning rate of 0.0001. The model is trained for a maximum of 200 iterations with a batch size of 2 per GPU.
In the training log, we can see that the model was trained for 200 iterations using the train dataset. The training progress was saved in the tutorial folder. During training, the model achieved an mIoU of 63.04% and an average accuracy of 84.29%. The per-class results showed that the background class achieved an IoU of 83.99% and an accuracy of 93.76%. The other three classes, namely "電車", "dc9", and "巴膠", achieved IoUs of 46.51%, 61.28%, and 60.4%, respectively, and accuracies of 70.91%, 97.21%, and 60.86%, respectively. The training log also shows the loss values and accuracy of the model at each iteration. These results provide an overview of the performance of the trained model on the validation set.

## Evaluation
The model is evaluated using the mean Intersection over Union (mIoU) metric. Evaluation is performed every 200 iterations during training, and the checkpoint with the best mIoU is saved.

## Usage
To use this model, please install the MMSegmentation package from https://github.com/open-mmlab/mmsegmentation and follow the instructions for running a segmentation model.

The checkpoint for this model is stored at checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth, and the working directory is set to ./work_dirs/tutorial.

This model was trained on a single GPU, so please adjust the gpu_ids parameter accordingly. The code also assumes that the dataset is stored in /content/drive/MyDrive/transport_segmentation.
