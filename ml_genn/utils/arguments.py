import argparse
from ml_genn.converters import rate_based
from ml_genn.layers import ConnectivityType

def parse_arguments(model_description='ML GeNN model'):
    '''
    Parses command line arguments for common ML GeNN options, and returns them in namespace form.
    '''

    parser = argparse.ArgumentParser(description=model_description)

    # compilation options
    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--rng-seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--few-spike', action='store_true')
    parser.add_argument('--input-type', default='poisson', 
                        choices=[i.value for i in rate_based.InputType])
    parser.add_argument('--connectivity-type', default='procedural', 
                        choices=[i.value for i in ConnectivityType])
    parser.add_argument('--kernel-profiling', action='store_true')

    # normalisation options
    parser.add_argument('--norm-method', default='data-norm', 
                        choices=[i.value for i in rate_based.NormMethod])
    parser.add_argument('--n-norm-samples', type=int, default=256)

    # evaluation options
    parser.add_argument('--n-train-samples', type=int, default=None)
    parser.add_argument('--n-test-samples', type=int, default=None)
    parser.add_argument('--save-samples', type=int, default=[], nargs='+')
    parser.add_argument('--plot', action='store_true')

    # TensorFlow options
    parser.add_argument('--reuse-tf-model', action='store_true')
    parser.add_argument('--record-tensorboard', action='store_true')
    parser.add_argument('--augment-training', action='store_true')

    return parser.parse_args()
