# Deep Convolutional Generative Adversial Network(DCGAN) for OASIS Brain data-set

### Description

The main idea of GAN is to generate new random images(fake images), which look more realistic than the input images. It consists of a Generator - to generate fake images, a Discriminator - to classify the image as a fake or real, and an adversarial network that pits them against each other. 

DCGAN is an extension of the GAN architecture with the use of convolutional neural networks in both the generator and discriminator parts. Implementation of the DCGAN model here follows the paper [Unsupervised Representation Learning With Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf) by Radford et. al. It suggests the constraints on the model required to develop the high-quality generator model effectively.

### Problem

Generating new brain images can be seen as a random variable generation problem, a probabilistic experiment. The sample image from the input OASIS brain dataset is shown below. It's 256 X 256 size with 1 grey-scale channel. Each image is a vector of 65,536-dimensions. We build a space with 65,536 axes, where each image will be a point in this space. Having a probability distribution function maps each input brain image to the non-negative real number and sums it to 1. GAN generates the new brain image by generating the new vector following this probability distribution over the 65,536-dimensional vector space, which is a very complex one and we don't know how to generate these complex random variables. The idea here is a transform method, generating 65,536 standard random normal variables(as a noise vector) and applying the complex function to this variable, where this complex functions are approximated by the CNN in the generator part and produces the 65,536-dimensional random variable that follows the input brain images probability distribution.


![](https://github.com/Pragatheeswari-dev/Deep_Learning/blob/main/Images/input_samples.png)

## Modeling

GanModel.py module has both the generator and discriminator network. Generator Networks take Input as a noise vector of shape (100,), generated from the standard normal distribution and in the early layer of the network it reshaped to the 65536. The remaining 5 layers consist of [convolutional-2DTranspose](https://naokishibuya.medium.com/up-sampling-with-transposed-convolution-9ae4f2df52d0) layer for upsampling the image to size (256,256) with stridded convolutions, Batch normalisation and Relu activation functions. Output Layer consists of tanh activation and produces the fake image of size (256,256,1). Using [LeakyRelu and elu activation](https://medium.com/@danqing/a-practical-guide-to-relu-b83ca804f1f7#:~:text=Leaky%20ReLU%20has%20a%20small,values%2C%20instead%20of%20altogether%20zero.&text=Parametric%20ReLU%20(PReLU)%20is%20a,%3D%20ax%20when%20x%20%3C%200) instead of [Relu and tanh activation](https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0) in generator network also produces better results.

Discriminator takes input images ( either real or fake) of size (256,256,1). It includes 6 convolution-2D layers with convolution strides for downsampling with LeakyRelu activation and uses drop layer with 0.3 for regularisation. The output layer uses sigmoid activation for classifying real(1) or fake(0) images.

Task-6 is the main file, it consists of a function to load the images(data_gen), a function to display the images produced by the generator network( display_images), other model parameters, building the models(generator, discriminator and combined networks), and finally training process for the number of epochs and plotting the training loss.

## Training

Training the GAN involves training the generator part to maximize the final binary classification error ( between real or fake image), while the discriminator is trained to minimize it. Reaches equilibrium when discriminator classifies with equal probability and the generator produces images that follow the input brain images probability distribution.

### Training procedure
The combined model is built by sequentially adding the generator and discriminator together. Compile the discriminator and set the training to false before compiling the combined model, so that only the generator's model parameters are updated. It is not necessary to compile the generator model as it's not run alone. 
1. Reading the real-images of Batch-size(16), using data-gen() from the OASIS Dataset.
1. Using a generator to generate fake images by giving input as random noise vectors for a batch size (16).
1. Train the discriminator with both real(1) and fake(0) images to classify. This will allow the discriminator's weights to be updated.
1. Train the combined model only using the fake images. Note when the combined model is trained only the generator's model parameter is updated.

## Results

Whole Training takes 20-25 minutes for 15000 Epochs to complete. Below is the Training loss for DCGAN and generated images for each epoch (mentioned on top), 4 noise vectors are given as input to the generator network and the corresponding 4 predicted image results are shown.

![](https://github.com/Pragatheeswari-dev/Deep_Learning/blob/main/Images/Training_loss_plot.jpg)

![](https://github.com/Pragatheeswari-dev/Deep_Learning/blob/main/Images/DCGAN_generator_images.jpg)
