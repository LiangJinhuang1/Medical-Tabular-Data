from train import train
from src.utils.config import load_config, get_config_value, resolve_path
from src.utils.save_utils import create_experiment_dir, save_configs
from src.utils.seed import set_seed
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run training multiple times with different excluded features')
    parser.add_argument('--exp-path', type=str, default='configs/__base__/exp_path.yaml', 
                       help='Experiment paths YAML')
    parser.add_argument('--train-args', type=str, default='configs/__base__/train_argument.yaml', 
                       help='Training arguments YAML')
    parser.add_argument('--train-file', type=str, default=None, 
                       help='Override: training CSV path')
    parser.add_argument('--exclude-cols', type=str, nargs='+', action='append', default=None,
                       help='Override: Lists of columns to exclude for each training run. Use multiple times: --exclude-cols col1 col2 --exclude-cols col3 col4 --exclude-cols col5 col6')
    parser.add_argument('--mlp-config', type=str, default=None, 
                       help='Override: MLP config YAML path')
    parser.add_argument('--tabm-config', type=str, default=None, 
                       help='Override: TabM config YAML path')
    parser.add_argument('--encoder-config', type=str, default=None, 
                       help='Override: Encoder config YAML path')
    parser.add_argument('--vae-encoder-config', type=str, default=None, 
                       help='Override: VAE Encoder config YAML path')
    
    args = parser.parse_args()
    
    # Load config files
    paths_cfg = load_config(args.exp_path)
    train_args = load_config(args.train_args)
    
    # Get target_col
    target_col = get_config_value(paths_cfg, 'paths', 'target_col')
    

    if args.exclude_cols:
        exclude_cols_list = args.exclude_cols
    else:
        exclude_cols_list = train_args.get('exclude_cols', [])

    
    
    # Prepare config paths 
    train_file = args.train_file or get_config_value(paths_cfg, 'paths', 'train_file')
    train_file = resolve_path(train_file)
    mlp_config_path = args.mlp_config or get_config_value(paths_cfg, 'paths', 'mlp_config')
    tabm_config_path = args.tabm_config or get_config_value(paths_cfg, 'paths', 'tabm_config')
    encoder_config_path = args.encoder_config or get_config_value(paths_cfg, 'paths', 'encoder_config')
    vae_encoder_config_path = args.vae_encoder_config or get_config_value(paths_cfg, 'paths', 'vae_encoder_config')
    
    # Load model configs
    mlp_config = load_config(mlp_config_path)
    tabm_config = load_config(tabm_config_path)
    encoder_model_cfg = load_config(encoder_config_path)
    
    mlp_model_cfg = mlp_config.get('model', {})
    tabm_model_cfg = tabm_config.get('model', {})
    encoder_model_cfg = encoder_model_cfg.get('model', {})
    training_cfg = train_args.get('training', {})
    seed = train_args.get('seed', 42)
    
    # Set seed for reproducibility
    set_seed(seed)
    
    # Create experiment directory and save config for this training run
    experiment_dir = create_experiment_dir(output_root=None)
    print(f'\nThe experiment output directory: {experiment_dir}')
    print(f'Target column: {target_col}')
        
    full_config = {
        'paths': {
            'train_file': str(train_file),
            'target_col': target_col,
            'mlp_config': str(mlp_config_path),
            'tabm_config': str(tabm_config_path),
        },
        'mlp_model': mlp_model_cfg,
        'tabm_model': tabm_model_cfg,
        'encoder_model': encoder_model_cfg,
        'training': training_cfg,
        'seed': seed,
        'exclude_cols': exclude_cols_list,
        'config_paths': {
            'exp_path': str(args.exp_path),
            'train_args': str(args.train_args),
            'mlp_config': str(mlp_config_path),
            'tabm_config': str(tabm_config_path),
            'encoder_config': str(encoder_config_path),
            'vae_encoder_config': str(vae_encoder_config_path),
        }
    }
    save_configs(experiment_dir, full_config)
    print(f'The config has been saved to: {experiment_dir / "configs"}')
        

    experiment_dir = train(
        target_col=target_col,
        exclude_cols=exclude_cols_list,
        train_file=train_file,
        train_args=train_args,
        mlp_model_cfg=mlp_model_cfg,
        tabm_model_cfg=tabm_model_cfg,
        encoder_model_cfg=encoder_model_cfg,
        training_cfg=training_cfg,
        experiment_dir=experiment_dir
    ) 
    print(f"\nResults saved to: {experiment_dir}")


if __name__ == '__main__':
    main()
