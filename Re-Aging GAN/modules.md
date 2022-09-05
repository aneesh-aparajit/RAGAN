# Modules present in the Code

## Overview

Let $\mathcal{X}$ and $\mathcal{Y}$ be the set of images and possible ages respectively. Given a face image $x\in\mathcal{X}$ and a target image $y'\in\mathcal{Y}$, our goal is to train a single Generator $G$ such that it can generate a face image $x'$ of a particular age $y'$ corresponding to the identity in $x$. In addition, we introduce an age modulator within G to reshape identity features by considering the target age and utilize it as a self-guiding information.

In comparison to prior works, the main objkect is to robustly transform face age as well as maximum retention of age-unrelated information in $x'$, such as background, hairstyle, expression, etc. This means this framework should preserve the age-unrelated information contained in the input image during the age transformation process. Therefore, we consider a simple strategey on encoder and decoder networtks to share some of the valuable information.

## Proposed Framework

The proposed GAN-based frame is divided into the generator consiting of an encoder, a modulator, and a decoder, and the discriminator. Since the discriminator follows exisiting approaches, it was not very broadly elaborated, only the generator part is described in depth. The generator makes use of encoder-decoder architecture for image generator and is made of an identity encoder $Enc$, an age modulator $AM$, and a decoder $Dec$. In a superficial view, the encoder and decoder networks perform the same procedyre as existing works, yet they have a few modifications.

One of the difference the proposed and the existing works is the indegration of addition sub-network at the bottleneck of the generator. By this networtk, we can obtain features providing information on how a particular person should look like at the age under consideration. Given that such age-ware features are learned based ona given input image, then it can be used for further generatio process.

## Identity Encoder

## Age Modulator

## Decoder

## Discriminator

## Optimization

## Experimental Setup