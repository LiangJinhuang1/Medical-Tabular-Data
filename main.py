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
        config = yaml.safe_load(f)
    return config


def main(args):
    # Parse arguments
    # load base configs
    base_paths = load_config(args.exp_path) if hasattr(args, 'exp_path') and args.exp_path else {}
    train_args = load_config(args.train_args) if hasattr(args, 'train_args') and args.train_args else {}

    # allow CLI override
    train_file = args.train_file or base_paths.get('train_file')
    val_file = args.val_file or base_paths.get('val_file')
    target_col = args.target_col or base_paths.get('target_col')
    mlp_config_path = args.mlp_config or base_paths.get('mlp_config')
    tabm_config_path = args.tabm_config or base_paths.get('tabm_config')
    
    # Load configurations
    mlp_config = load_config(mlp_config_path)
    tabm_config = load_config(tabm_config_path)
    
    # Get hyperparameters from config
    mlp_hidden_size = mlp_config['hidden_size']
    mlp_dropout = mlp_config['dropout']
    mlp_batchnorm = mlp_config['batchnorm']
    mlp_activation = mlp_config['activation']
    mlp_lr = mlp_config['learning_rate']
    mlp_epochs = mlp_config['epochs']
    batch_size = mlp_config['batch_size']
    
    tabm_hidden_size = tabm_config['hidden_size']
    tabm_k_heads = tabm_config['k_heads']
    tabm_adapter_dim = tabm_config['adapter_dim']
    tabm_dropout = tabm_config['dropout']
    tabm_lr = tabm_config['learning_rate']
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    try:
        # Load data
        print(f'Loading data from {train_file}')
        train_dataframe = load_data(train_file)
        print(f'Data Shape: {train_dataframe.shape}')

        print(f'Loading data from {val_file}')
        val_dataframe = load_data(val_file)
        print(f'Data Shape: {val_dataframe.shape}')

        # Create datasets
        train_dataset = Dataset(train_dataframe, target_col)
        val_dataset = Dataset(val_dataframe, target_col)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=train_args.get('num_workers', 0),
            pin_memory=train_args.get('pin_memory', False),
            prefetch_factor=train_args.get('prefetch_factor', None),
            persistent_workers=train_args.get('persistent_workers', False),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=train_args.get('num_workers', 0),
            pin_memory=train_args.get('pin_memory', False),
            prefetch_factor=train_args.get('prefetch_factor', None),
            persistent_workers=train_args.get('persistent_workers', False),
        )
        print(f'Created datasets: {len(train_dataset)} train, {len(val_dataset)} validation.')

        input_dim = train_dataframe.shape[1] - 1
        print(f'Input dimension: {input_dim}')

        # Create MLP model
        model_mlp = MLPRegressor(
            in_dim=input_dim,
            hidden_size=mlp_hidden_size,
            dropout=mlp_dropout,
            batchnorm=mlp_batchnorm,
            activation=mlp_activation
        ).to(device)
        print(f'\n----Simple MLP model ------')
        print(model_mlp)

        # Create TabM model
        model_tabm = TabM(
            in_dim=input_dim,
            out_dim=1,
            hidden_dims=tabm_hidden_size,
            k_heads=tabm_k_heads,
            adapter_dim=tabm_adapter_dim,
            dropout=tabm_dropout
        ).to(device)
        
        print(f'\n---- TabM model ------')
        print(f'k(ensemble size): {tabm_k_heads}')
        print(model_tabm)

        # Setup training
        loss_fn = MSELoss()
        optimizer_mlp = Adam(model_mlp.parameters(), lr=mlp_lr)
        optimizer_tabm = Adam(model_tabm.parameters(), lr=tabm_lr)
        
        print(f'\nStarting training...')
        print(f'Epoch | MLP train loss | MLP val loss | TabM train loss | TabM val loss')
        print('-' * 75)
        
        for epoch in range(mlp_epochs):
            train_loss_mlp = train_loop(model_mlp, train_loader, optimizer_mlp, loss_fn, device, model_type='mlp')
            train_loss_tabm = train_loop(model_tabm, train_loader, optimizer_tabm, loss_fn, device, model_type='tabm')
            val_loss_mlp = eval_loop(model_mlp, val_loader, loss_fn, device, model_type='mlp')
            val_loss_tabm = eval_loop(model_tabm, val_loader, loss_fn, device, model_type='tabm')
            
            print(f'{epoch + 1:5d}  | {train_loss_mlp:13.4f}  | {val_loss_mlp:11.4f} | {train_loss_tabm:13.4f}  | {val_loss_tabm:11.4f}')
        
        print(f'\nTraining complete!')
    
    except FileNotFoundError as e:
        print(f'\nExecution failed: {e}')
    except Exception as e:
        print(f'\nAn error occurred: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MLP and TabM models on medical tabular data')

    # Base config files
    parser.add_argument('--exp-path', type=str,
                        default='configs/__base__/exp_path.yaml',
                        help='Base experiment paths YAML')
    parser.add_argument('--train-args', type=str,
                        default='configs/__base__/train_argument.yaml',
                        help='Base training arguments YAML')

    # Optional CLI overrides
    parser.add_argument('--train-file', type=str, default=None, help='Override: training CSV path')
    parser.add_argument('--val-file', type=str, default=None, help='Override: validation CSV path')
    parser.add_argument('--target-col', type=str, default=None, help='Override: target column')
    parser.add_argument('--mlp-config', type=str, default=None, help='Override: MLP config YAML path')
    parser.add_argument('--tabm-config', type=str, default=None, help='Override: TabM config YAML path')

    args = parser.parse_args()
    main(args)

