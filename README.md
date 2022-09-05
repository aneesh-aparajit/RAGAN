# Re-Aging GAN: Towards Personalized Face Age Transformation

Face age transformation aims to synthesize past or future face images by reflecting the age factor on given faces. Ideally, this task should synthesize natural-looking faces across various age groups while maintaining identity. However, most of the existing work has focused on only one of these or is difficult to train while unnatural artifacts still appear.

Thus, the Re-Aging GAN (RAGAN) was proposed in ICCV 2021, a novel single framework considering all the critical factors in age transformation. Our framework achieves state-of-the-art personalized face age transformation by compelling the input identity to perform the self-guidance of the generation process. Specifically, RAGAN can learn the personalized age features by using high-order interactions between given identity and target age.

Our role in this project is to implement this paper, and try to replicate the results of the paper as much possible. The code with which they have implemented is neither open-sourced nor publically available.
