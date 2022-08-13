# MT-GAN
Based on the mutation operators, a series of mutated GANs is generated. Below we show 200 images. Except for the images generated by the original GAN in the first row, the rest of the images are generated by the mutated GANs. The second and third rows correspond to the hyper-parameter operator. Specifically, the second row corresponds to increasing the batch size to four times the original, and the third row corresponds to reducing the batch size to a quarter of the original. Rows four to nine all correspond to the noise injection operator, and we consider the three most common noises, exponential noise, gamma noise, and Gaussian noise. The fourth to sixth rows correspond to injecting exponential noise, gamma noise and Gaussian noise into all training images, respectively, and the seventh to ninth rows correspond to injecting exponential noise, gamma noise and Gaussian noise into one percent of the training images, respectively. The tenth row corresponds to the change sampling operator, see mnist_GAN_full_dataset_change_sampling.py for the specific implementation. Rows eleven to fifteen correspond to the conflicting discriminator operator. Specifically, the eleventh to thirteenth rows correspond to exchanging 50 pairs, 10 pairs, and 1 pair of data in one training round, respectively, and the fourteenth to fifteenth rows correspond to exchanging a pair of data every 100 rounds and one every 10,000 rounds, respectively. The sixteenth row corresponds to the randomness reduction operator, see mnist_GAN_full_dataset_rand_down.py for the specific implementation. Rows seventeen to nineteen correspond to injecting exponential noise, gamma noise, and Gaussian noise into five percent of the training images, respectively. The last row corresponds to changing the generator structure.

![Image text](https://github.com/Yuteng-Lu/MT-GAN/blob/main/generated_plot_e100.png)

![Image text](https://github.com/Yuteng-Lu/MT-GAN/blob/main/generated_plot_e100_2.png)


Next, we comb in detail the changes that different errors (mutations) bring to the GAN generation results. Through combing, we will summarize the image features corresponding to different mutations. 

Increasing the batch size may allow the feature information of different images to be finally aggregated into one generated image. For example, the third image in the second row shares the features of 3, 5, and 9. On the contrary, decreasing the batch size would keep the feature information from being captured by the generated images.

If there is obvious noise pollution in the images generated by GAN (corresponds to the fourth to sixth rows), then you can judge what kind of pollution is in the training data used based on the characteristics of the noise in the image (for example, the gamma noise is relatively more uneven). Note that when there is obvious pollution in the picture, a larger proportion of training pictures should be affected.

Rows seven to nine are characterized by lack or redundancy of strokes. For example, the first number in the eighth row is 6, but there is a redundant stroke extending from the bottom to the right; and the first number in the ninth row is 8, but its lower circle is not closed. This problem is due to the noise in some images polluting the space seen by the GAN, disrupting the learning of the GAN.

The erroneous feature on row ten results from changing the convolutional layer to upsampling, which suggests an inability to learn sufficiently smooth boundaries.

The effect of confusing discriminator (shown in the rows eleven to fifteen) is similar to contaminating the training data, which can bias the GAN's understanding of the sample space. This bias is manifested in the absence or redundancy of some segments in the generated images. Specific to the problem of generating numbers, we observed that confusing the discriminator makes it easier to change the topology of the generated image.

The generated images of random reduction (shown in the row sixteen) are characterized by some images with normal digits and plenty of the other ones with meaningless and messy lines. A GAN mutated by random reduction only guarantees the quality of the generated images which match the latent points used in training, which means that when we choose a latent point far away from each of them, the quality of the corresponding image is not guaranteed at all.

The generated images of five percent noise share the features displayed in those of one percent noise and full noise. Specifically, five percent noise could cause the generated images to have missing or redundant parts, and also cause some images to contain noisy points. Meanwhile, the images with noisy points are often in low quality.

Adding layers changes the model structure, and in essence, the mutated GAN becomes a new model. In this way, the original parameters will not be the optimal parameters for the new model. In fact, the effect is similar to that of the hyper-parameter operator, so that the features of the images in the twentieth row are similar to the features of the images in the third row. In practice, if you also encounter similar characteristics, please prioritize whether there is a problem with the parameters or the network structure.

In addition, it is worth pointing out that based on mutated GAN, we can also obtain some generalized adversarial examples. For example, the first image in the eleventh row is an adversarial example, which will be incorrectly identified as 2. The recognition result for all these two hundred images is [3 8 7 0 7 9 5 0 8 1, 9 0 9 0 9 9 8 9 0 7, 8 9 2 8 4 9 0 1 5 5, 9 7 9 6 8 7 3 6 9 1, 1 5 7 0 9 7 3 0 7 5, 3 2 7 9 3 4 0 8 8 2, 4 1 2 7 7 2 5 3 1 0, 6 7 3 4 7 9 0 0 7 1, 8 7 0 1 9 8 8 2 8 6, 7 1 8 5 7 3 2 3 0 0, 2 8 7 3 8 7 5 9 0 7, 7 9 3 9 8 2 0 3 8 1, 0 4 7 9 9 3 8 7 6 1, 2 3 4 9 0 6 9 7 0 6, 4 0 5 3 2 6 9 6 6 9, 7 8 3 1 2 0 1 3 8 6, 0 1 7 1 8 0 8 8 8 5, 3 8 7 0 7 1 0 2 0 7, 6 8 8 8 8 3 3 1 1 8, 9 6 5 0 4 0 2 0 7 4]. You can discover more adversarial examples based on this result.

In the future, we will further investigate how to draw more adversarial examples using mutated GAN. Note that the rationale for this thinking is that mutation perturbs the perception of space.  

