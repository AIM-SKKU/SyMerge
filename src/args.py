import os
import argparse

import torch

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--losstype",
        type=str,
        default="",
    )

    parser.add_argument('--trainlayer', nargs='+', type=int, help='training transformer layer')


    parser.add_argument(
        "--ood",
        type=str,
        default="",
    )

    parser.add_argument(
        "--twodataset",
        type=str,
        default="",
    )

    parser.add_argument(
        "--singledata",
        type=str,
        default="",
    )

    parser.add_argument(
        "--classifier_train",
        action="store_true",
        help="Enable classifier training"
    )

    parser.add_argument(
        "--adastart",
        action="store_true",
        help="coef init ; lw adamerging"
    )

    parser.add_argument(
        "--onlyclassifiertrain",
        action="store_true",
        help="coef init ; lw adamerging"
    )

    parser.add_argument(
        "--noclamp",
        action="store_true",
        help="disable lambda clamp"
    )

    parser.add_argument(
        "--randominitclassifier",
        action="store_true",
        help="classifier random initialization"
    )
    
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="sparse L1 constraints"
    )

    parser.add_argument(
        "--grad_mask",
        action="store_true",
        help="gradient mask"
    )

    parser.add_argument(
        "--surgery",
        action="store_true",
        help="representation surgery"
    )
    
    parser.add_argument(
        "--l1coef",
        type=float,  
        default=1e-4,  
        help="l1 coef"
    )

    parser.add_argument(
        "--prior",
        type=float,  
        default=0.3,  
        help="task arithmetic prior"
    )


    parser.add_argument(
        "--preweight",
        type=str,
        default="",
    )

    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. "
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only."
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--ls",
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        default='/gscratch/efml/gamaga/.cache/open_clip',
        help='Directory for caching models from OpenCLIP'
    )
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args
