from argparse import ArgumentParser
from functools import partial

from ml_genn.layers import InputType
from ml_genn.layers import ConnectivityType
from ml_genn.converters import ConverterType
from ml_genn.converters import Simple
from ml_genn.converters import DataNorm
from ml_genn.converters import SpikeNorm
from ml_genn.converters import FewSpike


def parse_arguments(model_description='ML GeNN model'):
    '''
    Parses command line arguments for common ML GeNN options, and returns them in namespace form.
    '''

    parser = ArgumentParser(description=model_description)

    # compilation options
    parser.add_argument('--dt', type=float, default=1.0)
    parser.add_argument('--rng-seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--input-type', default='poisson',
                        choices=[i.value for i in InputType])
    parser.add_argument('--connectivity-type', default='procedural',
                        choices=[i.value for i in ConnectivityType])
    parser.add_argument('--kernel-profiling', action='store_true')

    # ANN conversion options
    parser.add_argument('--converter', default='few-spike',
                        choices=[i.value for i in ConverterType])
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

    args = parser.parse_args()

    def build_converter(self, norm_data, signed_input=False, K=8, norm_time=500):
        if self.converter == 'few-spike':
            return FewSpike(K=K, signed_input=signed_input, norm_data=[norm_data])
        elif args.converter == 'data-norm':
            return DataNorm(norm_data=[norm_data], signed_input=signed_input, 
                            input_type=self.input_type)
        elif args.converter == 'spike-norm':
            return SpikeNorm(norm_data=[norm_data], norm_time=norm_time, 
                             signed_input=signed_input, input_type=self.input_type)
        else:
            return Simple(signed_input=signed_input, input_type=self.input_type)

    args.build_converter = partial(build_converter, args)

    return args
