import argparse
import yaml
import torch
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils.data import DataLoader

from src.models.TabM.tabM import TabM
from src.data.load_data import load_data
from src.models.MLP import MLPRegressor
from src.data.Dataset import Dataset
from src.training.train_loop import train_loop
from src.eval.eval_loop import eval_loop


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_config_value(config, *keys, default=None):
    """Helper to safely get nested config values"""
    for key in keys:
        config = config.get(key, {})
    return config if isinstance(config, dict) else (config if config is not None else default)


def create_dataloader(dataset, batch_size, shuffle, config):
    """Create DataLoader with config"""
    loader_cfg = config.get('loader', {})
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=loader_cfg.get('num_workers', 0),
        pin_memory=loader_cfg.get('pin_memory', False),
        prefetch_factor=loader_cfg.get('prefetch_factor', None),
        persistent_workers=loader_cfg.get('persistent_workers', False),
    )


def main(args):
    paths_cfg = load_config(args.exp_path)
    train_args = load_config(args.train_args)
    
    # Get paths with CLI override
    train_file = args.train_file or get_config_value(paths_cfg, 'paths', 'train_file')
    val_file = args.val_file or get_config_value(paths_cfg, 'paths', 'val_file')
    target_col = args.target_col or get_config_value(paths_cfg, 'paths', 'target_col')
    mlp_config_path = args.mlp_config or get_config_value(paths_cfg, 'paths', 'mlp_config')
    tabm_config_path = args.tabm_config or get_config_value(paths_cfg, 'paths', 'tabm_config')
    
    mlp_config = load_config(mlp_config_path)
    tabm_config = load_config(tabm_config_path)
    
    mlp_model_cfg = mlp_config.get('model', {})
    tabm_model_cfg = tabm_config.get('model', {})
    training_cfg = train_args.get('training', {})
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    try:
        print(f'Loading data from {train_file}')
        train_dataframe = load_data(train_file)
        print(f'Data Shape: {train_dataframe.shape}')

        print(f'Loading data from {val_file}')
        val_dataframe = load_data(val_file)
        print(f'Data Shape: {val_dataframe.shape}')

        train_dataset = Dataset(train_dataframe, target_col)
        val_dataset = Dataset(val_dataframe, target_col)

        train_loader = create_dataloader(train_dataset, training_cfg.get('batch_size', 64), True, train_args)
        val_loader = create_dataloader(val_dataset, training_cfg.get('batch_size', 64), False, train_args)
        print(f'Created datasets: {len(train_dataset)} train, {len(val_dataset)} validation.')

        input_dim = train_dataframe.shape[1] - 1
        print(f'Input dimension: {input_dim}')

        model_mlp = MLPRegressor(
            in_dim=input_dim,
            hidden_size=mlp_model_cfg.get('hidden_size', [128, 64]),
            dropout=mlp_model_cfg.get('dropout', 0.3),
            batchnorm=mlp_model_cfg.get('batchnorm', True),
            activation=mlp_model_cfg.get('activation', 'ReLU')
        ).to(device)
        print(f'\n---- Simple MLP model ------')
        print(model_mlp)

        model_tabm = TabM(
            in_dim=input_dim,
            out_dim=1,
            hidden_dims=tabm_model_cfg.get('hidden_size', [128, 128]),
            k_heads=tabm_model_cfg.get('k_heads', 8),
            adapter_dim=tabm_model_cfg.get('adapter_dim', None),
            dropout=tabm_model_cfg.get('dropout', 0.1)
        ).to(device)
        
        print(f'\n---- TabM model (k={tabm_model_cfg.get("k_heads", 8)}) ------')
        print(model_tabm)

        lr = training_cfg.get('learning_rate', 1e-3)
        loss_fn = MSELoss()
        optimizer_mlp = Adam(model_mlp.parameters(), lr=lr)
        optimizer_tabm = Adam(model_tabm.parameters(), lr=lr)
        
        print(f'\nStarting training...')
        print(f'Epoch | MLP train loss | MLP val loss | TabM train loss | TabM val loss')
        print('-' * 75)
        
        for epoch in range(training_cfg.get('epochs', 10)):
            train_loss_mlp = train_loop(model_mlp, train_loader, optimizer_mlp, loss_fn, device, model_type='mlp')
            train_loss_tabm = train_loop(model_tabm, train_loader, optimizer_tabm, loss_fn, device, model_type='tabm')
            val_loss_mlp = eval_loop(model_mlp, val_loader, loss_fn, device, model_type='mlp')
            val_loss_tabm = eval_loop(model_tabm, val_loader, loss_fn, device, model_type='tabm')
            
            print(f'{epoch + 1:5d}  | {train_loss_mlp:13.4f}  | {val_loss_mlp:11.4f} | {train_loss_tabm:13.4f}  | {val_loss_tabm:11.4f}')
        
        print(f'\nTraining complete!')
    
    except Exception as e:
        print(f'Error: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MLP and TabM models on medical tabular data')
    
    parser.add_argument('--exp-path', type=str, default='configs/__base__/exp_path.yaml', help='Experiment paths YAML')
    parser.add_argument('--train-args', type=str, default='configs/__base__/train_argument.yaml', help='Training arguments YAML')
    parser.add_argument('--train-file', type=str, default=None, help='Override: training CSV path')
    parser.add_argument('--val-file', type=str, default=None, help='Override: validation CSV path')
    parser.add_argument('--target-col', type=str, default=None, help='Override: target column')
    parser.add_argument('--mlp-config', type=str, default=None, help='Override: MLP config YAML path')
    parser.add_argument('--tabm-config', type=str, default=None, help='Override: TabM config YAML path')
    
    args = parser.parse_args()
    main(args)
