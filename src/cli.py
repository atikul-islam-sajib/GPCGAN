import sys
import argparse

sys.path.append("src/")

from data_loader import Loader
from generator import Generator
from discriminator import Discriminator
from trainer import Trainer
from test import Test

"""
# Main CLI for GAN Operations

This script serves as the main command-line interface (CLI) for various operations related to Generative Adversarial Networks (GANs). It integrates functionalities such as data loading, model training, and synthetic data generation.

## Features:
- Argument parsing for flexible configuration of operations like data loading, training, and synthetic image generation.
- Facilitates downloading and loading of MNIST dataset.
- Initiates the training of GAN models.
- Generates synthetic images using a trained generator.

## Usage:
Run the script from the command line with the desired arguments. For example:
    python main_cli.py --download_mnist --batch_size 32 --epochs 100 --latent_space 100 --lr 0.0002 --samples 20
    
Run the script from the command line for synthetic with the desired arguments. For example:
    python main_cli.py --samples 20 --latent_space 100 --test
    
    
## Arguments:
- `--batch_size`: Batch size for the DataLoader.
- `--download_mnist`: Flag to download the MNIST dataset.
- `--epochs`: Number of epochs for training.
- `--latent_space`: Dimension of the latent space for the generator.
- `--lr`: Learning rate for the optimizer.
- `--samples`: Number of synthetic samples to generate.
"""


def cli():
    """
    The main command-line interface (CLI) function for handling user input and coordinating the data loading, training, and synthetic image generation processes.

    ### Process:
    - Parses command-line arguments.
    - Based on the arguments, it either downloads and processes the MNIST dataset and/or trains the GAN models.
    - Generates synthetic images if specified in the arguments.

    ### Raises:
    - Exception: If the provided arguments do not meet the required conditions for the operations.
    """
    parser = argparse.ArgumentParser(description="Command line coding".title())
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for the DataLoader".capitalize(),
    )
    parser.add_argument(
        "--download_mnist",
        action="store_true",
        help="Download Mnist dataset".capitalize(),
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs".capitalize()
    )
    parser.add_argument(
        "--latent_space", type=int, default=100, help="Latent size".capitalize()
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Learning rate".capitalize()
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of samples to generate".capitalize(),
    )
    parser.add_argument(
        "--test", action="store_true", help="Run synthetic data tests".capitalize()
    )

    args = parser.parse_args()

    if args.download_mnist:
        if args.batch_size > 10 and args.epochs and args.latent_space > 50 and args.lr:
            loader = Loader(batch_size=args.batch_size)
            loader.create_loader(mnist_data=loader.download_mnist())

            trainer = Trainer(
                latent_space=args.latent_space, epochs=args.epochs, lr=args.lr
            )
            trainer.train_CGAN()
        else:
            raise Exception("Provide the arguments appropriate way".capitalize())

    if args.test:
        if args.samples % 2 == 0 and args.latent_space > 50:
            test = Test(num_samples=args.samples, latent_space=args.latent_space)
            test.plot_synthetic_image()
        else:
            raise Exception(
                "Please enter a valid number of samples and latent space".capitalize()
            )


if __name__ == "__main__":
    cli()
