import argparse
import logging
import sys
import torch
import torch.nn as nn
from collections import OrderedDict

sys.path.append("src/")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    filemode="w",
    filename="./logs/generator.log/",
)


class Generator(nn.Module):
    """
    A generator class for a Generative Adversarial Network (GAN), particularly used for
    generating images. It takes a latent space vector and a label as input and generates
    images corresponding to the input label. It utilizes fully connected layers and
    LeakyReLU activation for intermediate layers, with a Tanh activation for the output layer.

    Attributes:
        latent_space (int): Dimensionality of the latent space vector (z), which is a random
                            noise input for the generator.
        num_labels (int): Number of unique labels for the conditional GAN. It corresponds to
                          the number of different classes in the dataset.
        labels (nn.Embedding): Embedding layer for the labels, allowing the generator to use
                               label information to generate images corresponding to specific classes.
        layers_config (list): A list defining the architecture of the neural network. Each
                              element in the list is a tuple, with the first two elements
                              being the number of input and output features for a layer,
                              and the optional third being the negative slope for LeakyReLU.
        model (nn.Sequential): The actual neural network model, constructed based on layers_config.
                               It comprises fully connected (Linear) layers, LeakyReLU activation
                               for non-linearity in intermediate layers, and a Tanh activation function
                               in the output layer for generating pixel values.

    Methods:
        connected_layer(layers_config=None):
            Constructs the neural network layers based on layers_config. It initializes the fully
            connected layers and the activation functions, specifically using LeakyReLU for intermediate
            layers and Tanh for the output layer.

        forward(x, labels):
            Performs a forward pass of the generator. It takes a latent space vector `x` and its
            corresponding labels, processes them through the network, and generates a batch of images.

    Note:
        - The latent space vector x should be of the shape (N, latent_space) where N is the batch size.
        - Labels should be of shape (N,) and contain integers representing class labels.
        - The output is a tensor of shape (N, 1, 28, 28), representing generated images of size 28x28 pixels.
    """

    def __init__(self, latent_space=100, num_labels=10):
        self.latent_space = latent_space
        self.num_labels = num_labels
        super(Generator, self).__init__()
        self.labels = nn.Embedding(self.num_labels, self.num_labels)
        self.layers_config = [
            (self.latent_space + self.num_labels, 256, 0.2),
            (256, 512, 0.2),
            (512, 1024, 0.2),
            (1024, 784),
        ]
        self.model = self.connected_layer(layers_config=self.layers_config)

    def connected_layer(self, layers_config=None):
        layers = OrderedDict()
        if layers_config is not None:
            for index, (in_features, out_features, negative_slope) in enumerate(
                layers_config[:-1]
            ):
                layers["{}_layer".format(index)] = nn.Linear(
                    in_features=in_features, out_features=out_features
                )
                layers["{}_activation".format(index)] = nn.LeakyReLU(
                    negative_slope=negative_slope
                )

            (in_features, out_features) = layers_config[-1]
            layers["output_layer"] = nn.Linear(
                in_features=in_features, out_features=out_features
            )
            layers["output_activation"] = nn.Tanh()

            return nn.Sequential(layers)

        else:
            raise Exception("No layers config provided".capitalize())

    def forward(self, x, labels):
        if x is not None:
            labels = labels.long()
            labels = self.labels(labels)
            x = torch.cat([x, labels], dim=1)
            x = self.model(x)
        else:
            raise Exception("No input provided in Generator".capitalize())

        return x.reshape(-1, 1, 28, 28)


if __name__ == "__main__":

    def total_params(model=None):
        """
        Calculates the total number of parameters in a given PyTorch model.

        The function iterates over all parameters in the model and sums their number of elements to get the total parameter count.

        ### Parameters:
        - `model` (torch.nn.Module, optional): The model for which the total number of parameters is to be calculated.

        ### Returns:
        - `total_params` (int): The total number of parameters in the model.

        ### Raises:
        - Exception: If the model is not defined properly (i.e., `model` is None).
        """
        total_params = 0
        if model is not None:
            for _, params in model.named_parameters():
                total_params += params.numel()
        else:
            raise Exception("Model is not defined properly".capitalize())

        return total_params

    parser = argparse.ArgumentParser(
        description="Generator script for the MNIST dataset".title()
    )
    parser.add_argument(
        "--latent_space",
        type=int,
        default=100,
        help="Latent size for dataset".capitalize(),
    )
    parser.add_argument(
        "--labels",
        type=int,
        default=10,
        help="Labels size for dataset".capitalize(),
    )
    parser.add_argument(
        "--generator", action="store_true", help="Generator model".capitalize()
    )

    args = parser.parse_args()

    if args.generator:
        if args.labels > 1:
            generator = Generator(
                num_labels=args.labels, latent_space=args.latent_space
            )
            params = total_params(model=generator)
            logging.info(f"Total number of parameters: {params}".capitalize())
        else:
            raise Exception("Labels size must be greater than 1".capitalize())
    else:
        raise Exception("Generator model is not defined".capitalize())
