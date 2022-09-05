# Literature Survey

A face age transformation task is dedicated to learning age progression or regression of a given face occording to the target age. Here, the target age implies an explicit conditioning factor that guides the transformation to produce facial images with a certain age. That is, we can set any target age for an input face image, and expect to have an output face depicting the target age characteristics, as shown in the previous slide.

Ideally, age transformation models should satisfy the following properties.
1. A model should take into account the identity of the person while progressing/regressing the face age and systeain it mostly unaltered, _i.e._, identity preservation.
2. A model should be able to generate natural looking faces corresponding to the target age across various age group.

In this regard, a number of works on face age transformation have been introduced. These methods, on the basis of powerful generative adversarial networks (GANs). train deep neural networks to perform a robust age transformation of the input face. Aside from this, several mechanisms have been adopted (_i.e._ networks and constraits) for identity preservation to ensure that face identity is unaltered during the age transformation process. However, even with improved approaches, existing methods tend to generate images with visible artifacts and/or unnatural-looking faces which surely lowers down the image quality and its perception.

Another important aspect that should be considered is a wide-range age transformation,  specifically, an age regression process for rejuventating input face. Most existing works scarcely emphasise on this. Although a few methods can operate on face age regression and provide good performance, their results still suffer from artifacts near and/or on face regions and contain no background due to the tight face cropping.

The main contributions of this RAGAN are:
- They introduce a personalized self-guidance scheme that enables transforming the input face across various target age groups while preserving identity.
- They successfully perform age transformation using only a single generating and discriminative model trained effectively in an unpaired manner.
- They quantitatively and qualitatively demosntrate the superiority of RAGAN over state-of-the-art methods through extensive experimnents and evalutations.