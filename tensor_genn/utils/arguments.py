import argparse
from tensor_genn.layers import InputType, ConnectionType

def parse_arguments(model_description='Tensor GeNN model'):
    '''
    Parses command line arguments for common Tensor GeNN options, and returns them in namespace form.
    '''

    parser = argparse.ArgumentParser(description=model_description)

    # compilation options
    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--rng-seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--share-weights', action='store_true')
    parser.add_argument('--input-type', default='poisson', choices=[i.value for i in InputType])
    parser.add_argument('--connection-type', default='procedural', choices=[i.value for i in ConnectionType])

    # normalisation options
    parser.add_argument('--norm-method', default=None, choices=['data-norm', 'spike-norm'])
    parser.add_argument('--n-norm-samples', type=int, default=256)

    # evaluation options
    parser.add_argument('--classify-time', type=float, default=500.0)
    parser.add_argument('--n-train-samples', type=int, default=None)
    parser.add_argument('--n-test-samples', type=int, default=None)
    parser.add_argument('--save-samples', type=int, default=[], nargs='+')
    parser.add_argument('--plot', action='store_true')

    # TensorFlow options
    parser.add_argument('--reuse-tf-model', action='store_true')
    parser.add_argument('--record-tensorboard', action='store_true')
    parser.add_argument('--augment-training', action='store_true')

    return parser.parse_args()
