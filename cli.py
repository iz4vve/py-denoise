"""
Denoising module for wav files.

This module is designed to run on *.wav files, it will break if run on
other formats.
"""
import binascii
import os
import sys

import docopt
import wave

import numpy as np
from scipy import signal

assert sys.version_info >= (3, 6), "Python 3.6+ required for this script"

from denoise import *

__author__ = "Pietro Mascolo"
__version__ = "0.1"
DEFAULT_WINDOW = "hamming"


def denoise():
    """
    denoise removes low and high frequency band from
    a wav file and reports the result
    """
    usage = f"""Denoise.

        Usage:
        denoise.py --input=<noisy> --output=<denoised> [--low=<low>] [--high=<high>] [--window=<window>] [--cutoff=<cutoff>] [--algo=<algo>]
        denoise.py (-h | --help)

        Options:
        -h --help               Show this screen.
        --input=<noisy>         Path to audio file to denoise.
        --output=<denoised>     Path to denoised audio file.
        --window=<window>       Algorithm/technique for denoising [default: {DEFAULT_WINDOW}].
        --cutoff=<cutoff>       Cutoff frequency for the filter (Hz) [default: 400].
        --algo=<algo>           Filtering technique to use [default: window_filter]
    """
    OPERATORS = {
        "window_filter": window_filter,
    }
    arguments = docopt.docopt(usage, version=f"denoise {__version__}")

    algorithm = arguments.get("--algo")
    if not algorithm:
        print(
            f"Unsupported algorithm. Available techniques: f{OPERATORS.keys()}"
        )

    operator = OPERATORS[arguments["--algo"]]
    operator(arguments)




if __name__ == "__main__":
    denoise()
