# Re-Aging GAN

This is an attempt to try to implement the Re-Aging GANs (the face filter type). 

The main framework for this implementation is `PyTorch`.

## Resources

1. Lifespan Age Transformation Synthesis (LATS)
  - Link: https://arxiv.org/pdf/2003.09764.pdf
2. PFA-GAN: Progressive Face Aging with Generative Adversarial Network
  - Link: https://arxiv.org/pdf/2012.03459.pdf
3. Re-Aging GAN: Toward Personalized Face Age Transformation
  - Link: https://openaccess.thecvf.com/content/ICCV2021/papers/Makhmudkhujaev_Re-Aging_GAN_Toward_Personalized_Face_Age_Transformation_ICCV_2021_paper.pdf
4. Age Gap Reducer-GAN for Recognizing Age-Separated Faces
  - Link: https://arxiv.org/pdf/2011.05897.pdf
  
## Architecture Details

### Discriminator

- The discriminator is basically a [Pix2Pix](https://arxiv.org/pdf/1611.07004.pdf) disciminator, which follows a `PatchGAN` type structure.
- Inorder to make the model be aware of the input and the output ages, what we do is build an `Embedding` layer for the input and output ages where we encode each age group of the input dataset.
  - Here, we do a smaller version of the same, so we have 3 age categories. Although, we could extend that to a larger categorical size as well.
- If the image dimensions are `(BATCH_SIZE, 3, 224, 224)`, we build an Embedding layer with the `embedding_dim` as $224^2$. So that, we could eventually reshape the embedding to a `(BATCH_SIZE, 1, 224, 224)` and concatenate to the image at the $1^{\text{st}}$ axis, leading to an image dimension of `(BATCH_SIZE, 4, 224, 224)`.
- We do that same for the output image and then do the same as we would do for the Pix2Pix, concatenate both the images and pass it through the model.

#### Architecture Used for Discriminator:

```python
Discriminator(
  (inital): Sequential(
    (0): Conv2d(8, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): LeakyReLU(negative_slope=0.2)
  )
  (model): Sequential(
    (0): ConvBlock(
      (conv): Sequential(
        (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
    )
    (1): ConvBlock(
      (conv): Sequential(
        (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
    )
    (2): ConvBlock(
      (conv): Sequential(
        (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
    )
    (3): Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1))
  )
  (input_embed): Embedding(3, 65536)
  (output_embed): Embedding(3, 65536)
)
```


### Generator

- The generator is also inspired from the [Pix2Pix](https://arxiv.org/pdf/1611.07004.pdf), which inspired from the [U-Net: Convolutional Networks for Biomedical
Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf).
- I do the exact same thing as what I did to the discriminator. But, there is a small difference. I concatenate both the input and the output age with the input image. Which would result in an image shape of `(BATCH_SIZE, 5, 224, 224)`. 
- Then I simply pass it through the U-Net architecture.

#### Generator Model used

```python
Generator(
  (input_embed): Embedding(3, 65536)
  (output_embed): Embedding(3, 65536)
  (init_down): ConvBlock(
    (conv): Sequential(
      (0): Conv2d(5, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
  (down1): ConvBlock(
    (conv): Sequential(
      (0): Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
  (down2): ConvBlock(
    (conv): Sequential(
      (0): Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
  (down3): ConvBlock(
    (conv): Sequential(
      (0): Conv2d(256, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
  (down4): ConvBlock(
    (conv): Sequential(
      (0): Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
  (down5): ConvBlock(
    (conv): Sequential(
      (0): Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
  (down6): ConvBlock(
    (conv): Sequential(
      (0): Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
  (bottle_neck): Sequential(
    (0): Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), padding_mode=reflect)
    (1): LeakyReLU(negative_slope=0.2)
  )
  (up1): TransposeConvBlock(
    (tran_conv): Sequential(
      (0): ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
  (up2): TransposeConvBlock(
    (tran_conv): Sequential(
      (0): ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
  (up3): TransposeConvBlock(
    (tran_conv): Sequential(
      (0): ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
  (up4): TransposeConvBlock(
    (tran_conv): Sequential(
      (0): ConvTranspose2d(1024, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
  (up5): TransposeConvBlock(
    (tran_conv): Sequential(
      (0): ConvTranspose2d(1024, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
  (up6): TransposeConvBlock(
    (tran_conv): Sequential(
      (0): ConvTranspose2d(512, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
  (up7): TransposeConvBlock(
    (tran_conv): Sequential(
      (0): ConvTranspose2d(256, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
      (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )
  )
  (final_up): Sequential(
    (0): ConvTranspose2d(128, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    (1): Tanh()
  )
)
```

### Losses required to consider

The framework operates on three input information, an input image $x$ and its corresponding age label $y$, and randomly sampled target age $y'$ into which input should be transformed. Subsequently, given this information, G will produce __age-transformed__ $x'$, __reconstructed__ $x_{rec}$, and __cycle-consistency images__ $x_{cycle}$ as 

$$x'=G(x,y')$$

$$x_{rec}=G(x,y)$$

$$x_{cycle}=G(x',y)$$

#### Reconstruction loss

This loss monitors the case where the input and output ages are the same. Ideally, we expect the model to give the same image as the output which is monitored by the L1-Loss.

$$\mathcal{L}_{rec}(G) = \|x-x_{rec}\|_1$$

#### Cycle-Consistency loss

We have the age transformed image and let's say that the new input age is the ouptut age and the new output age is the input age. In such a case as well, we would want the output image to be as similar to the input image. This is also determined by the L1-Loss.

$$\mathcal{L}_{cyc}(G) = \|x-x_{cycle}\|_1$$

#### Adversarial loss

In general, GANs follow a Zero Sum Min-Max problem, so we use the standard GAN loss as well.

$$\mathcal{L}_{adv}(G,D)=\mathbb{E}_{x,y}[\log D_y(x)] + \mathbb{E}_{x,y'}[\log(1-D_{y'}(x'))]$$
