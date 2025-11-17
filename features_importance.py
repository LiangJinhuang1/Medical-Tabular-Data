import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from torch.utils.data import random_split
from torch.nn import MSELoss
from typing import Dict, List, Optional
import pandas as pd
import shap

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
    """Create all models"""
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
    mlp_embedding = EncoderEmbedding(model_encoder, mlp_embedding_model)
    tabm_embedding = EncoderEmbedding(model_encoder, tabm_embedding_model)
    mlp_vae_embedding = VAEEncoderMLP(model_vae_encoder, mlp_vae_embedding_model, use_mu=True, use_log_var=False)
    tabm_vae_embedding = VAEEncoderMLP(model_vae_encoder, tabm_vae_embedding_model, use_mu=True, use_log_var=False)
    
    return {
        'mlp': model_mlp,
        'tabm': model_tabm,
        'mlp_embedding': mlp_embedding,
        'tabm_embedding': tabm_embedding,
        'mlp_vae_embedding': mlp_vae_embedding,
        'tabm_vae_embedding': tabm_vae_embedding
    }


def load_model_from_checkpoint(checkpoint_path, model_type, input_dim, config, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    models = create_models(input_dim, config, device)
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    model = models[model_type]
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def get_feature_names_from_dataframe(df, target_col):
    """Get feature names from dataframe"""
    feature_cols = [col for col in df.columns if col != target_col]
    return feature_cols


def compute_gradient_based_importance(model, loader, device, model_type: str, input_dim: int) -> np.ndarray:
    """Compute feature importance based on gradients"""
    model.eval()
    importance = np.zeros(input_dim)
    n_samples = 0
    
    loss_fn = MSELoss(reduction='sum')
    
    for x, labels in loader:
        x = x.to(device)
        labels = labels.to(device)
        
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)
        
        x.requires_grad_(True)
        
        # Forward pass
        outputs = model(x)
        
        # Handle different model types
        if model_type == 'tabm' or 'tabm' in model_type:
            if outputs.dim() == 3:  # (batch, k_heads, out_dim)
                outputs = outputs.mean(dim=1)
        
        if outputs.dim() == 1:
            outputs = outputs.unsqueeze(1)
        
        # Compute loss
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Get gradients w.r.t. input
        if x.grad is not None:
            grad_importance = torch.abs(x.grad).mean(dim=0).cpu().numpy()
            importance += grad_importance * x.size(0)
            n_samples += x.size(0)
    
    if n_samples > 0:
        importance = importance / n_samples
    
    return importance


def compute_permutation_importance(model, loader, device, model_type: str, input_dim: int, 
                                   n_permutations: int = 5) -> np.ndarray:
    """Compute feature importance based on permutation (Permutation Importance)"""
    model.eval()
    
    # First compute baseline performance
    baseline_predictions = []
    baseline_labels = []
    
    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)
            labels = labels.to(device)
            
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
            
            outputs = model(x)
            
            if model_type == 'tabm' or 'tabm' in model_type:
                if outputs.dim() == 3:
                    outputs = outputs.mean(dim=1)
            
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            
            baseline_predictions.append(outputs.cpu().numpy())
            baseline_labels.append(labels.cpu().numpy())
    
    baseline_predictions = np.concatenate(baseline_predictions, axis=0).flatten()
    baseline_labels = np.concatenate(baseline_labels, axis=0).flatten()
    baseline_mse = np.mean((baseline_predictions - baseline_labels) ** 2)
    
    # Compute importance for each feature
    importance = np.zeros(input_dim)
    
    for feature_idx in range(input_dim):
        permuted_mses = []
        
        for _ in range(n_permutations):
            permuted_predictions = []
            permuted_labels = []
            
            with torch.no_grad():
                for x, labels in loader:
                    x = x.clone().to(device)
                    labels = labels.to(device)
                    
                    # Permute current feature
                    perm_indices = torch.randperm(x.size(0))
                    x[:, feature_idx] = x[perm_indices, feature_idx]
                    
                    if labels.dim() == 1:
                        labels = labels.unsqueeze(1)
                    
                    outputs = model(x)
                    
                    if model_type == 'tabm' or 'tabm' in model_type:
                        if outputs.dim() == 3:
                            outputs = outputs.mean(dim=1)
                    
                    if outputs.dim() == 1:
                        outputs = outputs.unsqueeze(1)
                    
                    permuted_predictions.append(outputs.cpu().numpy())
                    permuted_labels.append(labels.cpu().numpy())
            
            permuted_predictions = np.concatenate(permuted_predictions, axis=0).flatten()
            permuted_labels = np.concatenate(permuted_labels, axis=0).flatten()
            permuted_mse = np.mean((permuted_predictions - permuted_labels) ** 2)
            permuted_mses.append(permuted_mse)
        
        # Importance = permuted MSE - baseline MSE (larger difference means more important)
        importance[feature_idx] = np.mean(permuted_mses) - baseline_mse
    
    return importance


def compute_shap_importance(model, loader, device, model_type: str, 
                            feature_names: list, n_samples: int = 100) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    
    background_data = []
    sample_data = []
    total = 0
    
    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)
            
            background_data.append(x.cpu().numpy())
            sample_data.append(x.cpu().numpy())
            
            total += x.size(0)
            if total >= n_samples:
                break
    
    background_data = np.concatenate(background_data, axis=0)
    sample_data = np.concatenate(sample_data, axis=0)
    
    background_data = background_data[:n_samples]
    sample_data = sample_data[:n_samples]
    
    def model_wrapper(x):
        model.eval()
        with torch.no_grad():
            x_tensor = torch.as_tensor(x, dtype=torch.float32, device=device)
            outputs = model(x_tensor)
            
            if 'tabm' in model_type.lower() and outputs.dim() == 3:
                outputs = outputs.mean(dim=1)
            
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            
            return outputs.cpu().numpy()
    
    print(f'Computing SHAP using {background_data.shape[0]} background and {sample_data.shape[0]} samples...')
    
    explainer = shap.KernelExplainer(model_wrapper, background_data)
    shap_values = explainer.shap_values(sample_data, nsamples=min(100, len(sample_data)))
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    importance = np.abs(shap_values).mean(axis=0)  # (n_features,)
    
    # Ensure importance is 1D array
    importance = np.array(importance).flatten()
    
    # Ensure importance has the correct length
    if len(importance) != len(feature_names):
        print(f'Warning: SHAP importance length ({len(importance)}) does not match feature names length ({len(feature_names)})')
        # Truncate or pad to match feature_names length
        if len(importance) > len(feature_names):
            importance = importance[:len(feature_names)]
        else:
            importance = np.pad(importance, (0, len(feature_names) - len(importance)), 'constant')
    
    return importance


def plot_feature_importance(importance_scores: Dict[str, np.ndarray], 
                            feature_names: List[str],
                            save_path: Path,
                            top_n: int = 20):
    """Plot feature importance"""
    n_methods = len(importance_scores)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 10))
    
    if n_methods == 1:
        axes = [axes]
    
    for idx, (method_name, importance) in enumerate(importance_scores.items()):
        ax = axes[idx]
        
        # Sort and select top N
        sorted_indices = np.argsort(importance)[::-1][:top_n]
        sorted_importance = importance[sorted_indices]
        sorted_importance = np.array(sorted_importance).flatten().tolist()
        sorted_names = [feature_names[int(i)] for i in sorted_indices]
        
        # Plot bar chart
        bars = ax.barh(range(len(sorted_names)), sorted_importance, color='steelblue')
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names, fontsize=9)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'{method_name} - Top {top_n} Features', fontsize=14, fontweight='bold')
        ax.invert_yaxis()  # Most important at top
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, sorted_importance)):
            ax.text(val, i, f' {val:.4f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to: {save_path}")
    plt.close()


def save_importance_to_csv(importance_scores: Dict[str, np.ndarray],
                           feature_names: List[str],
                           save_path: Path):
    """Save feature importance to CSV file"""
    data = {'Feature_Name': feature_names}
    
    for method_name, importance in importance_scores.items():
        importance_1d = np.array(importance).flatten()
        data[method_name] = importance_1d
    
    df = pd.DataFrame(data)
    
    # Sort by importance for each method
    for method_name in importance_scores.keys():
        df_sorted = df.sort_values(by=method_name, ascending=False)
        csv_path = save_path.parent / f'{save_path.stem}_{method_name}.csv'
        df_sorted.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Feature importance CSV saved to: {csv_path}")


def find_available_models(checkpoints_dir):
    """Find all available model checkpoints"""
    available_models = []
    checkpoint_files = list(checkpoints_dir.glob('*_best.pt'))
    
    for checkpoint_file in checkpoint_files:
        model_name = checkpoint_file.stem.replace('_best', '')
        if model_name not in ['encoder', 'vae_encoder']:
            available_models.append(model_name)
    
    return sorted(available_models)


def analyze_feature_importance(experiment_dir: Path,
                               model_name: Optional[str] = None,
                               methods: List[str] = ['gradient', 'permutation', 'shap'],
                               top_n: int = 20,
                               n_permutations: int = 5,
                               n_shap_samples: int = 100):

    # Ensure experiment_dir is a Path object
    experiment_dir = Path(experiment_dir)
    
    if not experiment_dir.exists():
        raise ValueError(f"Experiment directory does not exist: {experiment_dir}")
    
    if not experiment_dir.is_dir():
        raise ValueError(f"Specified path is not a directory: {experiment_dir}")
    
    print(f'\n{"="*80}')
    print(f'Starting analysis for experiment: {experiment_dir.name}')
    print(f'Experiment directory: {experiment_dir}')
    print(f'{"="*80}\n')
    
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
    
    # Display experiment information
    print(f'Target Column: {target_col}')
    print(f'Training data file: {train_file}')
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    print(f'Loading data from {train_file}')
    full_dataframe = load_data(train_file)
    print(f'Data shape: {full_dataframe.shape}')
    
    # Handle excluded columns (consistent with training)
    exclude_cols = full_config.get('exclude_cols', [])
    if exclude_cols:
        if isinstance(exclude_cols[0], list):
            exclude_cols = [col for sublist in exclude_cols for col in sublist]
        cols_to_exclude = [col for col in exclude_cols if col in full_dataframe.columns]
        if cols_to_exclude:
            full_dataframe = full_dataframe.drop(columns=cols_to_exclude)
            print(f'Excluded {len(cols_to_exclude)} columns: {cols_to_exclude}')
    
    # Use the same random split as during training
    training_cfg = train_args.get('training', {})
    apply_normalization = training_cfg.get('apply_normalization', train_args.get('apply_normalization', False))
    
    # Create dataset 
    temp_dataset = Dataset(full_dataframe, target_col, apply_normalization=apply_normalization)
    
    # Get feature names directly from dataset 
    if hasattr(temp_dataset, 'feature_names'):
        feature_names = temp_dataset.feature_names
        print(f'Retrieved {len(feature_names)} feature names from Dataset')
    else:
        # Fallback: simulate filtering process if feature_names not available (old Dataset version)
        print('Warning: Dataset does not have feature_names attribute, using fallback method')
        df_for_filtering = full_dataframe.copy()
        missing_threshold = 200
        null_counts = df_for_filtering.isnull().sum()
        features_cols = [col for col in df_for_filtering.columns if col != target_col]
        features_to_drop = null_counts[features_cols][null_counts >= missing_threshold].index.tolist()
        
        if features_to_drop:
            df_for_filtering = df_for_filtering.drop(columns=features_to_drop)
        df_for_filtering = df_for_filtering.dropna()
        feature_names = [col for col in df_for_filtering.columns if col != target_col]
    
    # Verify dimensions match
    actual_input_dim = temp_dataset.features.shape[1]
    if len(feature_names) != actual_input_dim:
        print(f'Warning: Feature count mismatch. Expected: {len(feature_names)}, Actual: {actual_input_dim}')
        feature_names = [f'Feature_{i}' for i in range(actual_input_dim)]
        print('Using generic feature names as fallback')
    
    
    split_size = float(training_cfg.get('split_size', 0.2))
    total_size = temp_dataset.__len__()
    train_size = int(total_size * (1 - split_size))
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(temp_dataset, [train_size, test_size])
    
    print(f'Train data dimension: {train_dataset.__len__()}')
    print(f'Test data dimension: {test_dataset.__len__()}')
    
    # Create data loader
    batch_size = int(training_cfg.get('batch_size', 64))
    test_loader = create_dataloader(test_dataset, batch_size, False, train_args)
    
    # Get input dimension
    input_dim = temp_dataset.features.shape[1]
    
    # Find available models
    checkpoints_dir = experiment_dir / 'checkpoints'
    available_models = find_available_models(checkpoints_dir)
    
    if not available_models:
        raise ValueError("No model checkpoints found in experiment directory")
    
    # If model name is specified, process only that model; otherwise process all models
    if model_name is not None:
        if model_name not in available_models:
            raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
        models_to_process = [model_name]
    else:
        models_to_process = available_models
    
    print(f'\nFound {len(models_to_process)} model(s) to process: {models_to_process}')
    
    # Analyze feature importance for each model
    results_dir = experiment_dir / 'feature_importance'
    results_dir.mkdir(exist_ok=True)
    
    for model_name in models_to_process:
        print(f'\n{"="*60}')
        print(f'Processing model: {model_name}')
        print(f'{"="*60}')
        
        checkpoint_path = checkpoints_dir / f'{model_name}_best.pt'
        if not checkpoint_path.exists():
            print(f'Warning: Checkpoint not found for {model_name}, skipping...')
            continue
        
        try:
            print(f'Loading model from {checkpoint_path}')
            model = load_model_from_checkpoint(checkpoint_path, model_name, input_dim, full_config, device)
            
            # Compute feature importance using different methods
            importance_scores = {}
            
            if 'gradient' in methods:
                print('Computing gradient-based feature importance...')
                gradient_importance = compute_gradient_based_importance(model, test_loader, device, model_name, input_dim)
                importance_scores['Gradient Method'] = gradient_importance
            
            if 'permutation' in methods:
                print(f'Computing permutation-based feature importance (n_permutations: {n_permutations})...')
                permutation_importance = compute_permutation_importance(
                    model, test_loader, device, model_name, input_dim, n_permutations
                )
                importance_scores['Permutation Method'] = permutation_importance
            
            if 'shap' in methods:
                print(f'Computing SHAP-based feature importance (n_samples: {n_shap_samples})...')
                try:
                    shap_importance, _ = compute_shap_importance(
                        model, test_loader, device, model_name, feature_names, n_shap_samples
                    )
                    importance_scores['SHAP Method'] = shap_importance
                except Exception as e:
                    print(f'Error computing SHAP importance: {e}')
                    print('Skipping SHAP method...')
                    import traceback
                    traceback.print_exc()
            
            # Normalize importance scores (to 0-1 range for easier comparison)
            for method_name in importance_scores.keys():
                scores = importance_scores[method_name]
                if scores.max() > scores.min():
                    importance_scores[method_name] = (scores - scores.min()) / (scores.max() - scores.min())
                else:
                    importance_scores[method_name] = scores
            
            # Plot feature importance
            plot_path = results_dir / f'{model_name}_feature_importance.png'
            plot_feature_importance(importance_scores, feature_names, plot_path, top_n)
            
            # Save to CSV
            #csv_path = results_dir / f'{model_name}_feature_importance.csv'
            #save_importance_to_csv(importance_scores, feature_names, csv_path)
            
            print(f'\nFeature importance analysis completed for model {model_name}!')
            
        except Exception as e:
            print(f'Error processing {model_name}: {e}')
            import traceback
            traceback.print_exc()
            print(f'Skipping {model_name}...')
            continue
    
    print(f'\n{"="*60}')
    print('Feature importance analysis completed for all models!')
    print(f'Results saved to: {results_dir}')
    print(f'{"="*60}')


def analyze_multiple_experiments(experiment_dirs: List[Path],
                                 model_name: Optional[str] = None,
                                 methods: List[str] = ['gradient', 'permutation', 'shap'],
                                 top_n: int = 20,
                                 n_permutations: int = 1,
                                 n_shap_samples: int = 100):
    """
    Batch analyze feature importance for multiple experiments
    
    Args:
        experiment_dirs: List of experiment output directories
        model_name: Model name (if None, analyze all models)
        methods: List of methods to use
        top_n: Show top N important features
        n_permutations: Number of permutations for permutation importance
    """
    print(f'\n{"="*80}')
    print(f'Starting batch analysis for {len(experiment_dirs)} experiments')
    print(f'{"="*80}\n')
    
    for idx, exp_dir in enumerate(experiment_dirs, 1):
        print(f'\n{"#"*80}')
        print(f'Experiment {idx}/{len(experiment_dirs)}: {exp_dir.name}')
        print(f'{"#"*80}\n')
        
        try:
            analyze_feature_importance(
                experiment_dir=exp_dir,
                model_name=model_name,
                methods=methods,
                top_n=top_n,
                n_permutations=n_permutations,
                n_shap_samples=n_shap_samples
            )
            print(f'\n✓ Experiment {idx} analysis completed\n')
        except Exception as e:
            print(f'\n✗ Experiment {idx} analysis failed: {e}\n')
            import traceback
            traceback.print_exc()
            continue
    
    print(f'\n{"="*80}')
    print(f'Batch analysis completed! Processed {len(experiment_dirs)} experiments')
    print(f'{"="*80}\n')


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Feature Importance Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Analyze single experiment
  python features_importance.py --experiment_dir output/20241105_001311
  
  # Analyze multiple experiments (space-separated)
  python features_importance.py --experiment_dirs output/exp1 output/exp2 output/exp3
  
  # Analyze specific model only
  python features_importance.py --experiment_dir output/exp1 --model mlp
  
  # Use specific methods
  python features_importance.py --experiment_dir output/exp1 --methods gradient permutation shap
        """
    )
    
    # Support single or multiple experiment directories
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--experiment_dir', type=str, default=None,
                       help='Single experiment output directory path (must contain checkpoints and configs subdirectories)')
    group.add_argument('--experiment_dirs', type=str, nargs='+', default=None,
                       help='Multiple experiment output directory paths (space-separated)')
    
    parser.add_argument('--model', type=str, default=None,
                       help='Model name (if not specified, process all available models)')
    parser.add_argument('--methods', type=str, nargs='+', 
                       default=['gradient', 'permutation', 'shap'],
                       choices=['gradient', 'permutation', 'shap'],
                       help='List of methods to use')
    parser.add_argument('--top_n', type=int, default=20,
                       help='Show top N important features')
    parser.add_argument('--n_permutations', type=int, default=10,
                       help='Number of permutations for permutation importance')
    parser.add_argument('--n_shap_samples', type=int, default=200,
                       help='Number of samples to use for SHAP computation')
    
    args = parser.parse_args()
    
    # Handle single or multiple experiment directories
    if args.experiment_dir:
        # Single experiment
        analyze_feature_importance(
            experiment_dir=Path(args.experiment_dir),
            model_name=args.model,
            methods=args.methods,
            top_n=args.top_n,
            n_permutations=args.n_permutations,
            n_shap_samples=args.n_shap_samples
        )
    elif args.experiment_dirs:
        # Multiple experiments
        experiment_dirs = [Path(d) for d in args.experiment_dirs]
        analyze_multiple_experiments(
            experiment_dirs=experiment_dirs,
            model_name=args.model,
            methods=args.methods,
            top_n=args.top_n,
            n_permutations=args.n_permutations,
            n_shap_samples=args.n_shap_samples
        )


if __name__ == '__main__':
    main()

