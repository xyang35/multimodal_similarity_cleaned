"""
Default configurations for model evaluation 
"""

from .base_config import BaseConfig
import argparse

class EvalConfig(BaseConfig):
    def __init__(self):
        super(EvalConfig, self).__init__()

        self.parser.add_argument('--model_path', type=str, default=None,
                help='absolute path of pretrained model (including snapshot number')
        self.parser.add_argument('--sensors_path', type=str, default=None,
                help='absolute path of pretrained model (including snapshot number')
        self.parser.add_argument('--variable_name', type=str, default="",
                help='variable name for restoring model, e.g. modality_core')

        self.parser.add_argument('--feat', type=str, default='resnet',
                help='feature used')
        self.parser.add_argument('--network', type=str, default='tsn',
                help='Network used for sequence encoding')
        self.parser.add_argument('--preprocess_func', type=str, default='mean',
                help='Preprocessing function for input, ignored when model is defined: mean | max')
        self.parser.add_argument('--use_output', dest='use_output', action="store_true",
                help='Whether to use prediction output as embedding')
        self.parser.add_argument('--no_transfer', dest='transfer', action="store_false",
                help='Whether to transfer label')
        self.parser.set_defaults(transfer=True)

        self.parser.add_argument('--num_seg', type=int, default=3,
                       help='# of segment for a sequence')
        self.parser.add_argument('--emb_dim', type=int, default=256,
                       help='dimensionality of embedding')
        self.parser.add_argument('--batch_size', type=int, default=4,
                       help='Training batch size')
        self.parser.add_argument('--n_h', type=int, default=8,
                       help='height of the feature map')
        self.parser.add_argument('--n_w', type=int, default=8,
                       help='width of the feature map')
        self.parser.add_argument('--n_C', type=int, default=20,
                       help='number of output channels')
        self.parser.add_argument('--n_input', type=int, default=1536,
                       help='dim of input')

        self.parser.add_argument('--gpu', type=str, default=0,
                help='Set CUDA_VISIBLE_DEVICES')
        self.parser.add_argument('--label_type', type=str, default='goal',
                help='label_type: goal | stimuli')


        self.parser.add_argument('--no_normalized', dest='normalized', action="store_false",
                help='Whether embeddings are normalized to unit vector')
        self.parser.set_defaults(normalized=True)
        self.parser.add_argument('--reverse', dest='reverse', action="store_true",
                help='Whether to reverse input sequence')
        self.parser.set_defaults(reverse=False)


