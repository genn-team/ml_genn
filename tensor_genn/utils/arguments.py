import argparse
from tensor_genn import InputType

def parse_arguments(model_description='Tensor GeNN model'):
    '''
    Parses command line arguments for common Tensor GeNN options, and returns them in namespace form.
    '''
    parser = argparse.ArgumentParser(description=model_description)
    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--input-type', type=InputType, default=InputType.IF, choices=list(InputType))
    parser.add_argument('--rate-factor', type=float, default=1.0)
    parser.add_argument('--rng-seed', type=int, default=0)
    parser.add_argument('--norm-method', type=str, default=None, choices=['data-norm', 'spike-norm'])
    parser.add_argument('--classify-time', type=float, default=500.0)
    parser.add_argument('--n-train-samples', type=int, default=None)
    parser.add_argument('--n-test-samples', type=int, default=None)
    parser.add_argument('--n-norm-samples', type=int, default=256)
    parser.add_argument('--save-samples', type=int, default=[], nargs='+')
    parser.add_argument('--plot', action='store_true')
    return parser.parse_args()
