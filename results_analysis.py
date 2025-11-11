import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from torch.utils.data import random_split
from torch.nn import MSELoss

from src.data.load_data import load_data
from src.data.Dataset import Dataset
from src.data.dataloader import create_dataloader
from src.models.MLP import MLPRegressor
from src.models.TabM.tabM import TabM
from src.models.Embedding.EncoderMLP import EncoderEmbedding
from src.models.Embedding.VAEEncoderMLP import VAEEncoderMLP
from src.models.Embedding.TabAE import TabAE
from src.models.Embedding.TabVAE import TabVAE
from src.utils.config import load_config, get_config_value, resolve_path
from src.utils.seed import set_seed


def create_models(input_dim, config, device):
    mlp_model_cfg = config.get('mlp_model', {})
    tabm_model_cfg = config.get('tabm_model', {})
    encoder_model_cfg = config.get('encoder_model', {})
    
    # Create basic models
    model_mlp = MLPRegressor(
        in_dim=input_dim,
        hidden_size=mlp_model_cfg.get('hidden_size', [128, 64]),
        dropout=mlp_model_cfg.get('dropout', 0.3),
        batchnorm=mlp_model_cfg.get('batchnorm', True),
        activation=mlp_model_cfg.get('activation', 'ReLU')
    ).to(device)
    
    model_tabm = TabM(
        in_dim=input_dim,
        out_dim=1,
        hidden_dims=tabm_model_cfg.get('hidden_size', [128, 128]),
        k_heads=tabm_model_cfg.get('k_heads', 8),
        adapter_dim=tabm_model_cfg.get('adapter_dim', None),
        dropout=tabm_model_cfg.get('dropout', 0.1)
    ).to(device)
    
    # Create encoders
    encoder_latent_dim = encoder_model_cfg.get('latent_dim', 32)
    model_encoder = TabAE(
        input_dim=input_dim,
        latent_dim=encoder_latent_dim,
        hidden_dim=encoder_model_cfg.get('hidden_dim', [32, 12])
    ).to(device)
    
    model_vae_encoder = TabVAE(
        input_dim=input_dim,
        latent_dim=encoder_latent_dim,
        hidden_dim=encoder_model_cfg.get('hidden_dim', [32, 12])
    ).to(device)
    
    # Create embedding base models
    mlp_embedding_model = MLPRegressor(
        in_dim=encoder_latent_dim,
        hidden_size=mlp_model_cfg.get('hidden_size', [64, 32]),
        dropout=mlp_model_cfg.get('dropout', 0.3),
        batchnorm=mlp_model_cfg.get('batchnorm', True),
        activation=mlp_model_cfg.get('activation', 'ReLU')
    ).to(device)
    
    tabm_embedding_model = TabM(
        in_dim=encoder_latent_dim,
        out_dim=1,
        hidden_dims=tabm_model_cfg.get('hidden_size', [64, 64]),
        k_heads=tabm_model_cfg.get('k_heads', 8),
        adapter_dim=tabm_model_cfg.get('adapter_dim', None),
        dropout=tabm_model_cfg.get('dropout', 0.1)
    ).to(device)
    
    mlp_vae_embedding_model = MLPRegressor(
        in_dim=encoder_latent_dim,
        hidden_size=mlp_model_cfg.get('hidden_size', [64, 32]),
        dropout=mlp_model_cfg.get('dropout', 0.3),
        batchnorm=mlp_model_cfg.get('batchnorm', True),
        activation=mlp_model_cfg.get('activation', 'ReLU')
    ).to(device)
    
    tabm_vae_embedding_model = TabM(
        in_dim=encoder_latent_dim,
        out_dim=1,
        hidden_dims=tabm_model_cfg.get('hidden_size', [64, 64]),
        k_heads=tabm_model_cfg.get('k_heads', 8),
        adapter_dim=tabm_model_cfg.get('adapter_dim', None),
        dropout=tabm_model_cfg.get('dropout', 0.1)
    ).to(device)
    
    # Create combined embedding models 
    mlp_embedding = EncoderEmbedding(model_encoder, mlp_embedding_model, freeze_encoder=True)
    tabm_embedding = EncoderEmbedding(model_encoder, tabm_embedding_model, freeze_encoder=True)
    mlp_vae_embedding = VAEEncoderMLP(model_vae_encoder, mlp_vae_embedding_model, use_mu=True, use_log_var=False, freeze_encoder=True)
    tabm_vae_embedding = VAEEncoderMLP(model_vae_encoder, tabm_vae_embedding_model, use_mu=True, use_log_var=False, freeze_encoder=True)
    
    # Return dictionary mapping model names to models
    return {
        'mlp': model_mlp,
        'tabm': model_tabm,
        'mlp_embedding': mlp_embedding,
        'tabm_embedding': tabm_embedding,
        'mlp_vae_embedding': mlp_vae_embedding,
        'tabm_vae_embedding': tabm_vae_embedding
    }


def load_model_from_checkpoint(checkpoint_path, model_type, input_dim, config, device):
    """Load model checkpoint """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create all models 
    models = create_models(input_dim, config, device)
    
    # Get the specific model we need
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    model = models[model_type]
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def get_predictions_and_labels(model, loader, device, model_type):
    """Get all predictions and true labels"""
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)
            labels = labels.to(device)
            
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            
            # Forward pass
            outputs_raw = model(x)
            
            # Handle different model types
            if model_type == 'tabm' or 'tabm' in model_type:
                if outputs_raw.dim() == 3:  # (batch, k_heads, out_dim)
                    outputs_mean = outputs_raw.mean(dim=1)
                    outputs = outputs_mean
                else:
                    outputs = outputs_raw
            elif model_type == 'mlp' or 'embedding' in model_type:
                outputs = outputs_raw
            else:
                outputs = outputs_raw
            
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    all_predictions = all_predictions.flatten()
    all_labels = all_labels.flatten()
    
    return all_predictions, all_labels


def plot_error_analysis(true_values, predictions, model_name, save_path):
    """Plot error analysis: x-axis is True LVEF Value, y-axis is Residual (Error)"""
    residuals = predictions - true_values
    
    plt.figure(figsize=(10, 6))
    plt.scatter(true_values, residuals, alpha=0.5, s=20)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error Line')
    plt.xlabel('True LVEF Value', fontsize=12)
    plt.ylabel('Residual/Error', fontsize=12)
    plt.title(f'{model_name} - Error Analysis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add statistics
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals**2))
    mean_residual = np.mean(residuals)
    
    stats_text = f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nMean Residual: {mean_residual:.3f}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Error analysis plot saved to: {save_path}")
    plt.close()


def find_available_models(checkpoints_dir):
    """Find all available model checkpoints"""
    available_models = []
    checkpoint_files = list(checkpoints_dir.glob('*_best.pt'))
    
    for checkpoint_file in checkpoint_files:
        model_name = checkpoint_file.stem.replace('_best', '')
        if model_name not in ['encoder', 'vae_encoder']:
            available_models.append(model_name)
    
    return sorted(available_models)


def main(experiment_dir=None, model_name=None):
    # Determine experiment directory
    if experiment_dir is None:
        output_root = Path(__file__).parent / 'output'
        experiment_dirs = sorted([d for d in output_root.iterdir() if d.is_dir()], 
                                key=lambda x: x.name, reverse=True)
        if not experiment_dirs:
            raise ValueError("No experiment output directories found")
        experiment_dir = experiment_dirs[0] #last directory
    else:
        experiment_dir = Path(experiment_dir)
    
    if not experiment_dir.exists():
        raise ValueError(f"Experiment directory does not exist: {experiment_dir}")
    
    # Load configuration
    config_path = experiment_dir / 'configs' / 'full_config.yaml'
    if not config_path.exists():
        raise ValueError(f"Configuration file does not exist: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)
    
    # Load train_args configuration
    config_paths = full_config.get('config_paths', {})
    train_args_path = config_paths.get('train_args')
    if train_args_path:
        train_args = load_config(train_args_path)
    else:
        raise ValueError("train_args path not found in configuration")
    
    # Set random seed
    seed = train_args.get('seed', 42)
    set_seed(seed)
    
    # Get path configuration
    paths_cfg = full_config.get('paths', {})
    train_file = paths_cfg.get('train_file')
    target_col = paths_cfg.get('target_col', 'LVEF_dis')
    
    if train_file is None:
        raise ValueError("train_file path not found in configuration file")
    
    train_file = resolve_path(train_file)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print(f'Loading data from {train_file}')
    full_dataframe = load_data(train_file)
    print(f'Data Shape: {full_dataframe.shape}')

    # Use the same random split as during training
    training_cfg = train_args.get('training', {})
    apply_normalization = training_cfg.get('apply_normalization', train_args.get('apply_normalization', False))
    
    full_dataset = Dataset(full_dataframe, target_col, apply_normalization=apply_normalization)
    print(f'Data Dimension: {full_dataset.__len__()}')
    split_size = float(training_cfg.get('split_size', 0.2))
    total_size = full_dataset.__len__()
    train_size = int(total_size * (1 - split_size))
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    print(f'Train Data Dimension: {train_dataset.__len__()}')
    print(f'Test Data Dimension: {test_dataset.__len__()}')

    # Create data loader
    batch_size = int(training_cfg.get('batch_size', 64))
    test_loader = create_dataloader(test_dataset, batch_size, False, train_args)
    
    # Get input dimension
    input_dim = full_dataset.features.shape[1]
    
    # Find available models
    checkpoints_dir = experiment_dir / 'checkpoints'
    available_models = find_available_models(checkpoints_dir)
    
    if not available_models:
        raise ValueError("No model checkpoints found in the experiment directory")
    
    # If model_name is specified, use only that model; otherwise process all
    if model_name is not None:
        if model_name not in available_models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
        models_to_process = [model_name]
    else:
        models_to_process = available_models
    
    print(f'\nFound {len(models_to_process)} model(s) to process: {models_to_process}')
    
    # Plot error analysis for each model
    plots_dir = experiment_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    for model_name in models_to_process:
        print(f'\n{"="*60}')
        print(f'Processing model: {model_name}')
        print(f'{"="*60}')
        
        checkpoint_path = checkpoints_dir / f'{model_name}_best.pt'
        if not checkpoint_path.exists():
            print(f'Warning: Checkpoint not found for {model_name}, skipping...')
            continue
        
        try:
            print(f'Loading model from: {checkpoint_path}')
            model = load_model_from_checkpoint(checkpoint_path, model_name, input_dim, full_config, device)
            
            # Get predictions and true values
            print('Running evaluation...')
            predictions, true_values = get_predictions_and_labels(model, test_loader, device, model_name)
            
            print(f'Predictions shape: {predictions.shape}')
            print(f'Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]')
            print(f'True value range: [{true_values.min():.2f}, {true_values.max():.2f}]')
            
            # Plot error analysis
            save_path = plots_dir / f'{model_name}_error_analysis.png'
            plot_error_analysis(true_values, predictions, model_name.upper(), save_path)
            
        except Exception as e:
            print(f'Error processing {model_name}: {e}')
            print(f'Skipping {model_name}...')
            continue
    
    print(f'\n{"="*60}')
    print('Error analysis completed for all models!')
    print(f'{"="*60}')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Error Analysis: Plot residual plot')
    parser.add_argument('--experiment_dir', type=str, default=None,
                       help='Experiment output directory path (if not specified, use the latest output directory)')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name (if not specified, process all available models)')
    
    args = parser.parse_args()
    
    main(experiment_dir=args.experiment_dir, model_name=args.model)