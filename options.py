import argparse
import configparser
import json
import os
import time

import torch

from models.models import HierarchicalGraphNetModel, GraphUNetModel, GATConvModel, GCNConvModel


def get_options(args=None):
    init_parser = argparse.ArgumentParser(
        description="Hierarchical Graph ATtention (HGAT) networks",
        add_help=False
    )
    init_parser.add_argument('-m', '--model', default='GAT',
                        help="Graph Net Models: 'HGNet', 'GUNet', 'GAT', 'GCN'")
    init_parser.add_argument('-d', '--dataset', default='Cora',
                        help='PyG dataset name, e.g. "PubMed", "DD", "PROTEINS"')
    init_parser.add_argument('-c', '--conf_file', metavar="FILE",
                        help="Model & dataset config file")
    init_opts, _ = init_parser.parse_known_args(args)

    parser = argparse.ArgumentParser(
        parents=[init_parser],
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Select Model class
    model_class = {
        'HGNet': HierarchicalGraphNetModel,
        'GUNet': GraphUNetModel,
        'GAT': GATConvModel,
        'GCN': GCNConvModel
    }.get(init_opts.model, None)
    assert model_class is not None, f"Unknown model: {init_opts.model}"
    # Use the Model's staticmethod to specify its configuration options
    parser = model_class.add_config_arguments(parser)

    # Training options common to all models
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--fold', type=int, default=0,
                        help='Index of cross-validation fold for testing, set "-1" for no CV.' + \
                             'No CV available in transductive scenario (one graph).')
    parser.add_argument('--n_epochs', type=int, default=200, help='The number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=20, help='Size of a mini batch')
    parser.add_argument('--lr', type=float, default=0.005, help='Set the learning rate')
    parser.add_argument('--lr_decay', type=float, default=1.0, help='Learning rate decay per epoch')
    parser.add_argument('--wd', type=float, default=5e-4, help='Set parameter weight decay (L2 penalty)')

    # Run settings
    parser.add_argument('--run_name', default='run', help='Identifier for this experiment run')
    parser.add_argument('--tag', default='', help='Tag ID for this model config')
    parser.add_argument('--output_dir', default='outputs', help='Directory to store the output model')
    parser.add_argument('--log_step', type=int, default=0,
                        help='Log info every log_step steps, 0 for no intermediate logs')
    parser.add_argument('--log_dir', default='logs', help='Directory to write TensorBoard information to')
    parser.add_argument('--checkpoint_epochs', type=int, default=0,
                        help='Save checkpoint every n epochs, 0 to save no checkpoints')
    parser.add_argument('--time_limit', type=int, default=12,
                        help='Training loop will break after given number of hours; 0 = no limit')
    parser.add_argument('--no_tensorboard', action='store_true', help='Disable logging TensorBoard files')
    parser.add_argument('--no_progress_bar', action='store_true', help='Disable progress bar')
    parser.add_argument('--no_cuda', action='store_true', help='Do not use GPU')
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation only')
    parser.add_argument('--load_path', help='Path to load model parameters')

    # Load options from config file, these can be overridden by commandline args
    if not init_opts.conf_file:
        default_config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                           "configs",
                                           f"{init_opts.model}.conf")
        if os.path.isfile(default_config_file):
            init_opts.conf_file = default_config_file
        else:
            print("No config file given or found.\n")
    if init_opts.conf_file:
        config = configparser.SafeConfigParser()
        config.read([init_opts.conf_file])
        # Select model configuration for the given dataset
        selected_section = None
        for section in config.sections():
            if init_opts.dataset in section.split('|'):
                assert selected_section is None, \
                    f"Multiple sections for {init_opts.dataset} found in {init_opts.conf_file}"
                selected_section = section
        assert selected_section is not None, \
            f"Section for {init_opts.dataset} not found in {init_opts.conf_file}"
        opts_from_config = dict(config.items(selected_section))
        opts_from_config['conf_file'] = init_opts.conf_file
        parser.set_defaults(**opts_from_config)

    # Parse all commandline arguments one last time
    # the precedence of settings: commandline args > config file > defaults
    opts = parser.parse_args(args)

    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    fold_string = "" if opts.fold == -1 else f"_fold-{opts.fold}"
    model_string = opts.model if not opts.tag else f"{opts.model}-{opts.tag}"
    opts.run_name = f"{model_string}_{opts.run_name}{fold_string}_{time.strftime('%Y%m%dT%H%M%S')}"
    opts.save_dir = os.path.join(opts.output_dir, opts.dataset, opts.run_name)

    os.makedirs(opts.save_dir)
    # Save configuration of this run
    with open(os.path.join(opts.save_dir, "options.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    opts.model_class = model_class  # Already parsed above
    return opts
