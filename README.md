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
