import argparse

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Train MLP and TabM models on medical tabular data')
    parser.add_argument('--exp-path', type=str, default='configs/__base__/exp_path.yaml', help='Experiment paths YAML')
    parser.add_argument('--train-args', type=str, default='configs/__base__/train_argument.yaml', help='Training arguments YAML')
    parser.add_argument('--train-file', type=str, default=None, help='Override: training CSV path')
    parser.add_argument('--val-file', type=str, default=None, help='Override: validation CSV path')
    parser.add_argument('--target-col', type=str, default=None, help='Override: target column')
    parser.add_argument('--mlp-config', type=str, default=None, help='Override: MLP config YAML path')
    parser.add_argument('--tabm-config', type=str, default=None, help='Override: TabM config YAML path')
    parser.add_argument('--encoder-config', type=str, default=None, help='Override: Encoder config YAML path')
    parser.add_argument('--vae-encoder-config', type=str, default=None, help='Override: VAE Encoder config YAML path')
    return parser