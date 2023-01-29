from argparse import ArgumentParser
from sdne_lightning.train import train_sdne

import sys


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("input_graph")
    train_parser.add_argument("model_path")
    options = parser.parse_args(sys.argv[1:])

    if options.subcommand == "train":
        train_sdne(
            input_graph=options.input_graph, model_path=options.model_path
        )
