import torch
import torch.nn as nn
import torch.nn.functional as F
import typing


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2) -> None:
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=stride, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)



class Discriminator(nn.Module):
    '''Implementation is the same as Pix2Pix

    The forward method will have the the input image and the target image.
        - The target image may not be of the same person, so as result, we'll consider a few more loss functions like the reconstruction loss.
    '''
    def __init__(self, in_channels: int = 3, features: tuple = (64, 128, 256, 512), input_size: int = 128, num_age_groups: int = 3) -> None:
        super(Discriminator, self).__init__()
        self.inital = nn.Sequential(
            nn.Conv2d(in_channels=in_channels*2 + 2, out_channels=features[0], kernel_size=4, stride=2, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(negative_slope=0.2)
        )

        layers = []
        in_channels = features[0]

        for ix in range(1, len(features), 1):
            if ix == len(features) - 1:
                layers.append(ConvBlock(in_channels=in_channels, out_channels=features[ix], stride=1))
            else:
                layers.append(ConvBlock(in_channels=in_channels, out_channels=features[ix]))
            in_channels = features[ix]

        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'))

        self.model = nn.Sequential(*layers)

        self.input_size = input_size
        
        self.input_embed = nn.Embedding(num_embeddings=num_age_groups, embedding_dim=self.input_size ** 2)
        self.output_embed = nn.Embedding(num_embeddings=num_age_groups, embedding_dim=self.input_size ** 2)

    
    def forward(self, input_tensor: typing.Tuple[torch.Tensor, torch.Tensor], output_tensor: typing.Tuple[torch.Tensor, torch.Tensor], _print: bool = False) -> torch.Tensor:
        '''This is the forward call of the Discriminator
        
        Args:
        ----
            - input_tensor : Tuple[torch.Tensor, torch.Tensor]
                - The first element is the input image
                - The second element is the input age

            - output_tensor : Tuple[torch.Tensor, torch.Tensor]
                - The first element is the output image
                - The second element is the output age

        The considered solution, is similar to the original "Conditional Paper".
            - In this implementation, what we do is, we pass the input age embedding, output image, output age embedding.
                - The idea behind this is that the model will eventually learn the correlation between the pixel values in the input age, output image and the age groups.
                - Here, we can use a single age embedding layer, but I want to experiment with two different embedding layers of the input age and the output age.

        '''

        X_img, X_age = input_tensor
        y_img, y_age = output_tensor

        if _print:
            print(f'X_img: {X_img.shape}, X_age: {X_age.shape}')
            print(f'y_img: {y_img.shape}, y_age: {y_age.shape}')

        X_age_embed = self.input_embed(X_age).reshape(-1, 1, self.input_size, self.input_size)
        y_age_embed = self.output_embed(y_age).reshape(-1, 1, self.input_size, self.input_size)

        if _print:
            print(f'X_age_embed: {X_age_embed.shape}, y_age_embed: {y_age_embed.shape}')

        X = torch.cat([X_img, X_age_embed], dim=1)
        y = torch.cat([y_img, y_age_embed], dim=1)

        if _print:
            print(f'X: {X.shape}, y: {y.shape}')

        x = torch.cat([X, y], dim=1)

        if _print:
            print(f'x: {x.shape}')

        x = self.inital(x)

        if _print:
            print(f'x: {x.shape}')
        return self.model(x)


def test():
    X_img = torch.randn((1, 3, 256, 256))
    X_age = torch.tensor([[2]])

    y_img = torch.randn((1, 3, 256, 256))
    y_age = torch.tensor([[1]])
    
    model = Discriminator(input_size=256)

    print(model)


    print(f'\n[PROCESSING THE DATA...]\n')
    
    z = model.forward((X_img, X_age), (y_img, y_age), _print=True)

    print(f'z: {z.shape}')


if __name__ == '__main__':
    test()

