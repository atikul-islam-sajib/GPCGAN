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
    filename="./logs/discriminator.log/",
)


class Discriminator(nn.Module):
    """
    A discriminator class for a Generative Adversarial Network (GAN), particularly used for
    image data with an added conditional label embedding. The discriminator's goal is to
    differentiate between real and fake images. It utilizes fully connected layers,
    LeakyReLU activation for hidden layers, and a Sigmoid activation for the output layer.

    Attributes:
        num_labels (int): Number of unique labels for the conditional GAN. For instance,
                          in a dataset with 10 different classes, num_labels should be 10.
        labels (nn.Embedding): Embedding layer for the labels, which allows the discriminator
                               to condition the input on a particular class.
        layers_config (list): A list defining the architecture of the neural network. Each
                              element in the list is a tuple, with the first two elements
                              being the number of input and output features for a layer,
                              and the optional third being the negative slope for LeakyReLU.
        model (nn.Sequential): The actual neural network model, constructed based on layers_config.
                              It comprises fully connected (Linear) layers, LeakyReLU activation
                              for non-linearity in hidden layers, and a Sigmoid activation function
                              in the output layer to obtain probabilities.

    Methods:
        connected_layer(layers_config=None):
            Constructs the neural network layers based on layers_config. It initializes the fully
            connected layers and the activation functions, specifically using LeakyReLU for hidden
            layers and Sigmoid for the output layer.

        forward(x, labels):
            Performs a forward pass of the discriminator. It takes an input batch of images `x`
            and their corresponding labels, processes the images and labels through the network,
            and outputs a batch of probabilities indicating how likely each image is to be real.

    Note:
        - The model expects inputs x to be of the shape (N, 784) where N is the batch size.
        - Labels should be of shape (N,) and contain integers representing class labels.
        - The output is of shape (N, 1), representing the likelihood of each image being real.
        - The network architecture is designed to work with flattened images of size 28x28 pixels.
    """

    def __init__(self, num_labels=10):
        self.num_labels = num_labels
        super(Discriminator, self).__init__()
        self.labels = nn.Embedding(self.num_labels, self.num_labels)
        self.layers_config = [
            (784 + self.num_labels, 1024, 0.2, 0.3),
            (1024, 512, 0.2, 0.3),
            (512, 256, 0.2, 0.3),
            (256, 1),
        ]
        self.model = self.connected_layer(layers_config=self.layers_config)

    def connected_layer(self, layers_config=None):
        layers = OrderedDict()
        if layers_config is not None:
            for index, (
                in_features,
                out_features,
                negative_slope,
                dropout,
            ) in enumerate(layers_config[:-1]):
                layers["{}_layer".format(index + 1)] = nn.Linear(
                    in_features=in_features, out_features=out_features
                )
                layers["{}_activation".format(index + 1)] = nn.LeakyReLU(
                    negative_slope=negative_slope, inplace=True
                )
                layers["{}_dropout".format(index + 1)] = nn.Dropout(p=dropout)

            (in_features, out_features) = layers_config[-1]
            layers["output_layer"] = nn.Linear(
                in_features=in_features, out_features=out_features
            )
            layers["output_activation"] = nn.Sigmoid()

            return nn.Sequential(layers)
        else:
            raise Exception("Layers config is not defined properly".capitalize())

    def forward(self, x, labels):
        if x is not None:
            labels = labels.long()
            labels = self.labels(labels)
            x = x.reshape(-1, 28 * 28)
            x = torch.cat([x, labels], dim=1)
            x = self.model(x)
        else:
            raise Exception("Inputs are not defined properly".capitalize())

        return x


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
        description="Discriminator script for the MNIST dataset".title()
    )
    parser.add_argument(
        "--labels", type=int, default=10, help="Labels size for dataset".capitalize()
    )
    parser.add_argument(
        "--discriminator", action="store_true", help="Discriminator model".capitalize()
    )

    args = parser.parse_args()

    if args.discriminator:
        if args.labels > 1:
            discriminator = Discriminator(num_labels=args.labels)
            params = total_params(model=discriminator)
            logging.info(f"Total number of parameters: {params}".capitalize())
        else:
            raise Exception("Labels size must be greater than 1".capitalize())
    else:
        raise Exception("Discriminator model is not defined".capitalize())
