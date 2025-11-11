"""
Exploratory Data Analysis Script
Performs statistical analysis on training data, including:
1. Statistical information for each variable
2. Correlation heatmap (R²) between variables
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.load_data import load_data
from src.utils.config import load_config, get_config_value
from src.utils.arg_parser import build_arg_parser


def calculate_statistics(df):
    """Calculate statistical information for each variable"""
    stats = pd.DataFrame({
        'Variable': df.columns,
        'Data_Type': df.dtypes.values,
        'Non_Null_Count': df.count(),
        'Missing_Count': df.isnull().sum().values,
        'Missing_Percentage': (df.isnull().sum() / len(df) * 100).values,
        'Mean': df.mean().values,
        'Std': df.std().values,
        'Min': df.min().values,
        'Median': df.median().values,
        'Max': df.max().values,
    })
    
    # Round numerical columns to 2 decimal places
    numerical_cols = ['Missing_Percentage', 'Mean', 'Std', 'Min', 'Median', 'Max']
    for col in numerical_cols:
        if col in stats.columns:
            stats[col] = stats[col].round(2)
    
    return stats


def plot_correlation_heatmap(corr_matrix, output_path, figsize=(16, 14)):
    """Plot correlation coefficient heatmap (full matrix)"""
    plt.figure(figsize=figsize)
    
    # Show full matrix
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation (r)"},
        vmin=-1,
        vmax=1,
        annot_kws={'size': 6}
    )
    
    plt.title('Variable Correlation Heatmap (Correlation Coefficient)', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'Correlation heatmap saved to: {output_path}')


def plot_r2_heatmap(corr_matrix, output_path, figsize=(16, 14)):
    """Plot R² heatmap (full matrix)"""
    # Calculate R² (square of correlation coefficient)
    r_squared = corr_matrix ** 2
    
    plt.figure(figsize=figsize)
    
    # Show full matrix
    sns.heatmap(
        r_squared,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "R²"},
        vmin=0,
        vmax=1,
        annot_kws={'size': 6}
    )
    
    plt.title('Variable Correlation Heatmap (R²)', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'R² heatmap saved to: {output_path}')


def main():
    """Main function"""
    # Parse command line arguments
    parser = build_arg_parser()
    args = parser.parse_args()
    
    # Load configuration
    paths_cfg = load_config(args.exp_path)
    
    # Get data paths
    train_file = args.train_file or get_config_value(paths_cfg, 'paths', 'train_file')
    target_col = args.target_col or get_config_value(paths_cfg, 'paths', 'target_col')
    
    # Handle data paths - resolve relative paths
    if train_file:
        train_file = Path(train_file)
        if not train_file.is_absolute():
            # If --data is provided, use that directory
            if args.data:
                data_dir = Path(args.data)
                train_file = data_dir / train_file
            else:
                # Otherwise, resolve relative to project root
                train_file = project_root / train_file
        train_file = str(train_file)
    
    # Output directory
    output_dir = Path(__file__).parent  # Save to script directory
    
    print(f'=' * 60)
    print(f'Exploratory Data Analysis')
    print(f'=' * 60)
    print(f'Data file: {train_file}')
    print(f'Target column: {target_col}')
    print(f'Output directory: {output_dir}')
    print(f'=' * 60)
    
    try:
        # Load data
        print(f'\nLoading data...')
        full_dataframe = load_data(train_file)
        
        # Check if data was loaded successfully
        if full_dataframe is None:
            raise FileNotFoundError(f'Failed to load data from {train_file}. Please check if the file exists and the path is correct.')
        
        print(f'Data shape: {full_dataframe.shape}')
        print(f'Number of columns: {len(full_dataframe.columns)}')
        print(f'Number of features: {len(full_dataframe.columns) - 1}')  # Excluding target column
        print(f'\nUsing original dataset (no processing applied, all data)')
        
        # 1. Calculate statistics
        print(f'\nCalculating statistics...')
        stats = calculate_statistics(full_dataframe)
        
        # Save statistics
        stats_file = output_dir / 'statistical_summary.csv'
        stats.to_csv(stats_file, index=False, encoding='utf-8-sig')
        print(f'Statistics saved to: {stats_file}')
        
        # Print partial statistics to console
        print(f'\nStatistics summary:')
        print(stats.head(10).to_string())
        if len(stats) > 10:
            print(f'\n... (Total {len(stats)} variables, see CSV file for complete information)')
        
        # 2. Calculate correlation matrix
        print(f'\nCalculating correlation matrix...')
        # Only use numeric columns for correlation
        numeric_cols = full_dataframe.select_dtypes(include=[np.number]).columns
        corr_matrix = full_dataframe[numeric_cols].corr()
        
        # Save correlation matrix
        corr_file = output_dir / 'correlation_matrix.csv'
        corr_matrix.to_csv(corr_file, encoding='utf-8-sig')
        print(f'Correlation matrix saved to: {corr_file}')
        
        # 3. Plot correlation and R² heatmaps
        print(f'\nGenerating heatmaps...')
        
        # Adjust figure size if there are too many variables
        n_vars = len(corr_matrix)
        if n_vars > 50:
            figsize = (max(20, n_vars * 0.3), max(18, n_vars * 0.3))
        else:
            figsize = (16, 14)
        
        # Plot correlation coefficient heatmap (full matrix)
        corr_heatmap_file = output_dir / 'correlation_heatmap.png'
        plot_correlation_heatmap(corr_matrix, corr_heatmap_file, figsize=figsize)
        
        # Plot R² heatmap (full matrix)
        r2_heatmap_file = output_dir / 'correlation_heatmap_r2.png'
        plot_r2_heatmap(corr_matrix, r2_heatmap_file, figsize=figsize)
        
        # 4. Save R² matrix
        r_squared_matrix = corr_matrix ** 2
        r2_file = output_dir / 'r_squared_matrix.csv'
        r_squared_matrix.to_csv(r2_file, encoding='utf-8-sig')
        print(f'R² matrix saved to: {r2_file}')
        
        print(f'\n' + '=' * 60)
        print(f'Analysis complete! All results saved to: {output_dir}')
        print(f'Generated files:')
        print(f'  1. statistical_summary.csv - Statistical summary')
        print(f'  2. correlation_matrix.csv - Correlation coefficient matrix')
        print(f'  3. r_squared_matrix.csv - R² matrix')
        print(f'  4. correlation_heatmap.png - Correlation coefficient heatmap (full matrix)')
        print(f'  5. correlation_heatmap_r2.png - R² heatmap (full matrix)')
        print(f'=' * 60)
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

