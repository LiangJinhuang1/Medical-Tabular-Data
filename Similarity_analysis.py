import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from torch.utils.data import random_split
from typing import Dict, List, Optional, Tuple
import pandas as pd
from scipy.spatial.distance import euclidean, mahalanobis
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import random
import warnings
warnings.filterwarnings('ignore')

from src.data.load_data import load_data
from src.data.Dataset import Dataset
from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed

# Set font for Chinese characters (if needed)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def compute_euclidean_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance matrix"""
    return cdist(X, X, metric='euclidean')


def compute_cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix"""
    similarity = cosine_similarity(X)
    return similarity


def compute_mahalanobis_distance_matrix(X: np.ndarray) -> np.ndarray:
    """Compute Mahalanobis distance matrix"""
    n_samples, n_features = X.shape
    
    # Compute covariance matrix
    cov = np.cov(X.T)
    
    # If covariance matrix is singular, add small regularization term
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # If matrix is singular, use pseudo-inverse
        cov_inv = np.linalg.pinv(cov)
    
    # Compute mean
    mean = np.mean(X, axis=0)
    
    # Compute Mahalanobis distance matrix
    distances = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            diff = X[i] - X[j]
            distances[i, j] = np.sqrt(diff.T @ cov_inv @ diff)
    
    return distances


def find_k_nearest_neighbors(distances: np.ndarray, k: int, 
                            similarity_matrix: Optional[np.ndarray] = None) -> Dict[int, List[Tuple[int, float]]]:
    """
    Find k nearest neighbors for each patient
    
    Args:
        distances: Distance matrix (larger distance means lower similarity)
        k: Number of nearest neighbors
        similarity_matrix: Similarity matrix (optional, used for sorting if provided)
    
    Returns:
        Dictionary with patient indices as keys and list of [(neighbor_index, distance/similarity), ...] as values
    """
    n_samples = distances.shape[0]
    k = min(k, n_samples - 1)  # Ensure k does not exceed sample count - 1 (exclude self)
    
    neighbors_dict = {}
    
    for i in range(n_samples):
        # Get distances to all other patients
        dists = distances[i].copy()
        dists[i] = np.inf  # Exclude self
        
        # Find indices of k nearest neighbors
        nearest_indices = np.argsort(dists)[:k]
        nearest_distances = dists[nearest_indices]
        
        # Store results
        neighbors_dict[i] = [(int(idx), float(dist)) for idx, dist in zip(nearest_indices, nearest_distances)]
    
    return neighbors_dict


def plot_distance_distribution(distances: np.ndarray, title: str, save_path: Path):
    """Plot distance distribution"""
    # Only take upper triangle matrix (avoid duplicates)
    upper_triangle = np.triu(distances, k=1)
    distances_flat = upper_triangle[upper_triangle > 0]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(distances_flat, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Distance', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'{title} - Distance Distribution Histogram', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(distances_flat, vert=True)
    axes[1].set_ylabel('Distance', fontsize=12)
    axes[1].set_title(f'{title} - Distance Distribution Boxplot', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Distance distribution plot saved to: {save_path}")
    plt.close()


def plot_similarity_heatmap(similarity_matrix: np.ndarray, title: str, save_path: Path, 
                           max_samples: int = 100):
    """Plot similarity heatmap"""
    n_samples = similarity_matrix.shape[0]
    
    # If too many samples, only show first max_samples
    if n_samples > max_samples:
        print(f"Sample count ({n_samples}) exceeds {max_samples}, showing heatmap for first {max_samples} samples only")
        similarity_matrix = similarity_matrix[:max_samples, :max_samples]
        n_samples = max_samples
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(similarity_matrix, cmap='viridis', square=True, 
                cbar_kws={'label': 'Similarity'}, ax=ax)
    ax.set_title(f'{title} - Similarity Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Patient Index', fontsize=12)
    ax.set_ylabel('Patient Index', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Similarity heatmap saved to: {save_path}")
    plt.close()


def plot_distance_heatmap(distance_matrix: np.ndarray, title: str, save_path: Path, 
                          max_samples: int = 100):
    """Plot distance heatmap"""
    n_samples = distance_matrix.shape[0]
    
    # If too many samples, only show first max_samples
    if n_samples > max_samples:
        print(f"Sample count ({n_samples}) exceeds {max_samples}, showing heatmap for first {max_samples} samples only")
        distance_matrix = distance_matrix[:max_samples, :max_samples]
        n_samples = max_samples
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(distance_matrix, cmap='YlOrRd', square=True, 
                cbar_kws={'label': 'Distance'}, ax=ax)
    ax.set_title(f'{title} - Distance Heatmap', fontsize=14, fontweight='bold')
    ax.set_xlabel('Patient Index', fontsize=12)
    ax.set_ylabel('Patient Index', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Distance heatmap saved to: {save_path}")
    plt.close()


def plot_knn_results(knn_results: Dict[str, Dict[int, List[Tuple[int, float]]]], 
                     k: int, save_path: Path):
    """Plot k-NN results"""
    n_methods = len(knn_results)
    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 8))
    
    if n_methods == 1:
        axes = [axes]
    
    for idx, (method_name, neighbors_dict) in enumerate(knn_results.items()):
        ax = axes[idx]
        
        # Collect all distance/similarity values
        all_values = []
        for patient_idx, neighbors in neighbors_dict.items():
            for neighbor_idx, value in neighbors:
                all_values.append(value)
        
        # Plot distribution
        ax.hist(all_values, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel('Distance/Similarity', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{method_name} - k={k} Nearest Neighbor Distance Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_val = np.mean(all_values)
        median_val = np.median(all_values)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.4f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"k-NN results plot saved to: {save_path}")
    plt.close()


def plot_knn_examples(knn_results: Dict[str, Dict[int, List[Tuple[int, float]]]], 
                      test_features: np.ndarray, k: int, n_examples: int, 
                      save_path: Path, seed: Optional[int] = None):
    """
    Plot k-NN results for example patients
    
    Args:
        knn_results: Dictionary containing k-NN results for different methods
        test_features: Test set features
        k: Number of nearest neighbors
        n_examples: Number of example patients to plot
        save_path: Path to save the plot
        seed: Random seed for selecting example patients (optional)
    """
    n_methods = len(knn_results)
    n_examples = min(n_examples, len(test_features))
    
    # Randomly select example patients (with seed for reproducibility)
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    example_indices = np.random.choice(len(test_features), n_examples, replace=False)
    
    fig, axes = plt.subplots(n_examples, n_methods, 
                            figsize=(6 * n_methods, 4 * n_examples))
    
    if n_examples == 1:
        axes = axes.reshape(1, -1)
    if n_methods == 1:
        axes = axes.reshape(-1, 1)
    
    for row, patient_idx in enumerate(example_indices):
        for col, (method_name, neighbors_dict) in enumerate(knn_results.items()):
            ax = axes[row, col]
            
            # Get k nearest neighbors for this patient
            neighbors = neighbors_dict[patient_idx]
            neighbor_indices = [idx for idx, _ in neighbors]
            
            # Use PCA or first two features for visualization (if feature dimension > 2)
            if test_features.shape[1] > 2:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=2)
                features_2d = pca.fit_transform(test_features)
                explained_var = pca.explained_variance_ratio_.sum()
            else:
                features_2d = test_features
                explained_var = 1.0
            
            # Plot all patients (light color)
            ax.scatter(features_2d[:, 0], features_2d[:, 1], 
                      c='lightgray', s=20, alpha=0.5, label='All Patients')
            
            # Plot current patient (red)
            ax.scatter(features_2d[patient_idx, 0], features_2d[patient_idx, 1], 
                      c='red', s=200, marker='*', edgecolors='black', 
                      linewidths=2, label='Current Patient', zorder=5)
            
            # Plot k nearest neighbors (blue)
            neighbor_2d = features_2d[neighbor_indices]
            ax.scatter(neighbor_2d[:, 0], neighbor_2d[:, 1], 
                      c='blue', s=100, marker='o', edgecolors='black', 
                      linewidths=1.5, label=f'k={k} Nearest Neighbors', zorder=4)
            
            # Connection lines
            for neighbor_idx in neighbor_indices:
                ax.plot([features_2d[patient_idx, 0], features_2d[neighbor_idx, 0]],
                       [features_2d[patient_idx, 1], features_2d[neighbor_idx, 1]],
                       'b--', alpha=0.3, linewidth=1)
            
            ax.set_xlabel(f'Principal Component 1 ({explained_var*100:.1f}% variance explained)' if test_features.shape[1] > 2 else 'Feature 1', 
                         fontsize=10)
            ax.set_ylabel(f'Principal Component 2' if test_features.shape[1] > 2 else 'Feature 2', 
                         fontsize=10)
            ax.set_title(f'{method_name}\nPatient #{patient_idx}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"k-NN examples plot saved to: {save_path}")
    plt.close()


def save_knn_results_to_csv(knn_results: Dict[str, Dict[int, List[Tuple[int, float]]]], 
                            save_path: Path):
    """Save k-NN results to CSV"""
    for method_name, neighbors_dict in knn_results.items():
        rows = []
        for patient_idx, neighbors in neighbors_dict.items():
            for rank, (neighbor_idx, value) in enumerate(neighbors, 1):
                rows.append({
                    'Patient_Index': patient_idx,
                    'Neighbor_Rank': rank,
                    'Neighbor_Index': neighbor_idx,
                    'Distance_Similarity': value
                })
        
        df = pd.DataFrame(rows)
        csv_path = save_path.parent / f'{save_path.stem}_{method_name}.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"k-NN results CSV saved to: {csv_path}")


def analyze_patient_similarity(experiment_dir: Path,
                               k: int = 5,
                               max_samples_for_heatmap: int = 100,
                               n_examples: int = 3):
    """
    Analyze patient similarity
    
    Args:
        experiment_dir: Experiment output directory path
        k: k value for k-NN
        max_samples_for_heatmap: Maximum number of samples for heatmap
        n_examples: Number of example patients
    """
    experiment_dir = Path(experiment_dir)
    
    if not experiment_dir.exists():
        raise ValueError(f"Experiment directory does not exist: {experiment_dir}")
    
    print(f'\n{"="*80}')
    print(f'Starting patient similarity analysis: {experiment_dir.name}')
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
        raise ValueError("train_args path not found in configuration file")
    
    # Set random seed for reproducibility
    seed = train_args.get('seed', 42)
    print(f'Setting random seed to: {seed}')
    set_seed(seed)
    # Additional seed setting for sklearn and other libraries
    random.seed(seed)
    np.random.seed(seed)
    
    # Get path configuration
    paths_cfg = full_config.get('paths', {})
    train_file = paths_cfg.get('train_file')
    target_col = paths_cfg.get('target_col', 'LVEF_dis')
    
    if train_file is None:
        raise ValueError("train_file path not found in configuration file")
    
    train_file = resolve_path(train_file)
    
    print(f'Target column: {target_col}')
    print(f'Training data file: {train_file}')
    
    # Load data
    print(f'\nLoading data from {train_file}...')
    full_dataframe = load_data(train_file)
    print(f'Data shape: {full_dataframe.shape}')
    
    # Handle excluded columns
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
    apply_normalization = training_cfg.get('apply_normalization', 
                                          train_args.get('apply_normalization', False))
    
    # Create dataset
    full_dataset = Dataset(full_dataframe, target_col, apply_normalization=apply_normalization)
    
    split_size = float(training_cfg.get('split_size', 0.2))
    total_size = full_dataset.__len__()
    train_size = int(total_size * (1 - split_size))
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    print(f'Train data dimension: {train_dataset.__len__()}')
    print(f'Test data dimension: {test_dataset.__len__()}')
    
    # Get test set features
    test_indices = test_dataset.indices
    test_features = full_dataset.features[test_indices].numpy()
    test_labels = full_dataset.label[test_indices].numpy()
    
    print(f'\nTest set feature shape: {test_features.shape}')
    print(f'Test set label range: [{test_labels.min():.4f}, {test_labels.max():.4f}]')
    
    # Create results directory
    results_dir = experiment_dir / 'similarity_analysis'
    results_dir.mkdir(exist_ok=True)
    
    print(f'\n{"="*60}')
    print('Computing distance and similarity matrices...')
    print(f'{"="*60}')
    
    # Compute Euclidean distance
    print('Computing Euclidean distance matrix...')
    euclidean_distances = compute_euclidean_distance_matrix(test_features)
    print(f'Euclidean distance matrix shape: {euclidean_distances.shape}')
    print(f'Euclidean distance range: [{euclidean_distances.min():.4f}, {euclidean_distances.max():.4f}]')
    
    # Compute cosine similarity
    print('\nComputing cosine similarity matrix...')
    cosine_similarities = compute_cosine_similarity_matrix(test_features)
    print(f'Cosine similarity matrix shape: {cosine_similarities.shape}')
    print(f'Cosine similarity range: [{cosine_similarities.min():.4f}, {cosine_similarities.max():.4f}]')
    
    # Compute Mahalanobis distance
    print('\nComputing Mahalanobis distance matrix...')
    mahalanobis_distances = compute_mahalanobis_distance_matrix(test_features)
    print(f'Mahalanobis distance matrix shape: {mahalanobis_distances.shape}')
    print(f'Mahalanobis distance range: [{mahalanobis_distances.min():.4f}, {mahalanobis_distances.max():.4f}]')
    
    # Find k nearest neighbors
    print(f'\n{"="*60}')
    print(f'Finding k={k} nearest neighbors for each test patient...')
    print(f'{"="*60}')
    
    # For cosine similarity, convert to distance (1 - similarity) for k-NN
    cosine_distances = 1 - cosine_similarities
    
    knn_results = {
        'Euclidean Distance': find_k_nearest_neighbors(euclidean_distances, k),
        'Cosine Similarity': find_k_nearest_neighbors(cosine_distances, k),
        'Mahalanobis Distance': find_k_nearest_neighbors(mahalanobis_distances, k)
    }
    
    print(f'Completed k-NN search')
    
    # Plot results
    print(f'\n{"="*60}')
    print('Generating visualization plots...')
    print(f'{"="*60}')
    
    # 1. Distance distribution plots
    plot_distance_distribution(euclidean_distances, 'Euclidean Distance', 
                               results_dir / 'euclidean_distance_distribution.png')
    plot_distance_distribution(mahalanobis_distances, 'Mahalanobis Distance', 
                               results_dir / 'mahalanobis_distance_distribution.png')
    
    # 2. Similarity heatmap
    plot_similarity_heatmap(cosine_similarities, 'Cosine Similarity', 
                           results_dir / 'cosine_similarity_heatmap.png',
                           max_samples=max_samples_for_heatmap)
    
    # 2b. Distance heatmaps
    plot_distance_heatmap(euclidean_distances, 'Euclidean Distance', 
                          results_dir / 'euclidean_distance_heatmap.png',
                          max_samples=max_samples_for_heatmap)
    plot_distance_heatmap(mahalanobis_distances, 'Mahalanobis Distance', 
                          results_dir / 'mahalanobis_distance_heatmap.png',
                          max_samples=max_samples_for_heatmap)
    
    # 3. k-NN results distribution
    plot_knn_results(knn_results, k, results_dir / 'knn_distance_distribution.png')
    
    # 4. k-NN examples visualization
    plot_knn_examples(knn_results, test_features, k, n_examples, 
                     results_dir / 'knn_examples.png', seed=seed)
    
    # 5. Save k-NN results to CSV
    save_knn_results_to_csv(knn_results, results_dir / 'knn_results')
    
    # 6. Generate summary statistics
    print(f'\n{"="*60}')
    print('Generating summary statistics...')
    print(f'{"="*60}')
    
    summary_stats = {
        'Euclidean Distance': {
            'Mean': float(np.mean(euclidean_distances)),
            'Median': float(np.median(euclidean_distances)),
            'Std': float(np.std(euclidean_distances)),
            'Min': float(np.min(euclidean_distances)),
            'Max': float(np.max(euclidean_distances))
        },
        'Cosine Similarity': {
            'Mean': float(np.mean(cosine_similarities)),
            'Median': float(np.median(cosine_similarities)),
            'Std': float(np.std(cosine_similarities)),
            'Min': float(np.min(cosine_similarities)),
            'Max': float(np.max(cosine_similarities))
        },
        'Mahalanobis Distance': {
            'Mean': float(np.mean(mahalanobis_distances)),
            'Median': float(np.median(mahalanobis_distances)),
            'Std': float(np.std(mahalanobis_distances)),
            'Min': float(np.min(mahalanobis_distances)),
            'Max': float(np.max(mahalanobis_distances))
        }
    }
    
    summary_df = pd.DataFrame(summary_stats).T
    summary_path = results_dir / 'similarity_summary_statistics.csv'
    summary_df.to_csv(summary_path, encoding='utf-8-sig')
    print(f'Summary statistics saved to: {summary_path}')
    print('\nSummary Statistics:')
    print(summary_df)
    
    print(f'\n{"="*80}')
    print('Patient similarity analysis completed!')
    print(f'Results saved to: {results_dir}')
    print(f'{"="*80}\n')
    
    return results_dir


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Patient Similarity Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Analyze single experiment
  python Similarity_analysis.py --experiment_dir output/20241105_001311
  
  # Specify k value
  python Similarity_analysis.py --experiment_dir output/20241105_001311 --k 10
  
  # Specify maximum samples for heatmap
  python Similarity_analysis.py --experiment_dir output/20241105_001311 --max_samples 50
        """
    )
    
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Experiment output directory path (must contain checkpoints and configs subdirectories)')
    parser.add_argument('--k', type=int, default=5,
                       help='k value for k-NN (default: 5)')
    parser.add_argument('--max_samples', type=int, default=100,
                       help='Maximum number of samples for heatmap (default: 100)')
    parser.add_argument('--n_examples', type=int, default=3,
                       help='Number of example patients (default: 3)')
    
    args = parser.parse_args()
    
    analyze_patient_similarity(
        experiment_dir=Path(args.experiment_dir),
        k=args.k,
        max_samples_for_heatmap=args.max_samples,
        n_examples=args.n_examples
    )


if __name__ == '__main__':
    main()

