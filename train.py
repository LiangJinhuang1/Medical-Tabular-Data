import torch
from torch.utils.data import random_split
from torch.optim import Adam
from torch.nn import MSELoss
from src.data.dataloader import create_dataloader
from src.utils.save_utils import (
    save_checkpoint, 
    LossTracker
)
from src.models.TabM.tabM import TabM
from src.data.load_data import load_data
from src.models.MLP import MLPRegressor
from src.data.Dataset import Dataset
from src.training.train_loop import train_loop
from src.eval.eval_loop import eval_loop
from src.models.Embedding.EncoderMLP import EncoderEmbedding
from src.models.Embedding.TabAE import TabAE
from src.models.Embedding.TabVAE import TabVAE
from src.models.Embedding.VAEEncoderMLP import VAEEncoderMLP
from src.training.embedding_train_loop import train_encoder, train_vae_encoder


def train(target_col, exclude_cols=None, train_file=None, train_args=None, 
          mlp_model_cfg=None, tabm_model_cfg=None, encoder_model_cfg=None,
          training_cfg=None, experiment_dir=None):

    train_args = train_args 
    training_cfg = training_cfg
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    try:
        print(f'Loading data from {train_file}')
        full_dataframe = load_data(train_file)
        print(f'Data Shape: {full_dataframe.shape}')

        # Exclude columns from dataframe if specified
        if exclude_cols:
            cols_to_exclude = [col for col in exclude_cols if col in full_dataframe.columns]
            if cols_to_exclude:
                full_dataframe = full_dataframe.drop(columns=cols_to_exclude)
            else:
                print(f'Warning: None of the specified exclude_cols {exclude_cols} were found in the dataframe')

        # Get normalization option from training config
        apply_normalization = training_cfg.get('apply_normalization', train_args.get('apply_normalization', False))
        full_dataset = Dataset(full_dataframe, target_col, apply_normalization=apply_normalization)
        print(f'Data Dimension: {full_dataset.__len__()}')
        
        split_size = float(train_args.get('split_size', training_cfg.get('split_size', 0.2)))
        total_size = full_dataset.__len__()
        train_size = int(total_size * (1 - split_size))
        test_size = total_size - train_size
        train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])


        print(f'Train Data Dimension: {train_dataset.__len__()}')
        print(f'Test Data Dimension: {test_dataset.__len__()}')


        batch_size = int(training_cfg.get('batch_size', 64))
        train_loader = create_dataloader(train_dataset, batch_size, True, train_args)
        test_loader = create_dataloader(test_dataset, batch_size, False, train_args)
        print(f'Created datasets: {len(train_dataset)} train, {len(test_dataset)} test')

        # Get input dimension from the dataset
        input_dim = full_dataset.features.shape[1]
        print(f'Input dimension: {input_dim}')

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


        lr = float(training_cfg.get('learning_rate', 1e-3))
        weight_decay = float(training_cfg.get('weight_decay', 1e-5))
        loss_fn = MSELoss()


        # optimizers
        optimizer_mlp = Adam(model_mlp.parameters(), lr=lr)
        optimizer_tabm = Adam(model_tabm.parameters(), lr=lr)
        optimizer_encoder = Adam(model_encoder.parameters(), lr=lr,weight_decay=weight_decay)
        optimizer_vae_encoder = Adam(model_vae_encoder.parameters(), lr=lr,weight_decay=weight_decay)
        
        
        # initialize the loss tracker
        loss_tracker = LossTracker()
        
        num_epochs = int(training_cfg.get('epochs', 10))
        
        # Train the encoder 
        train_encoder(model_encoder, train_loader, optimizer_encoder, loss_fn, device, num_epochs, loss_tracker)

        # Train the VAE encoder
        train_vae_encoder(model_vae_encoder, train_loader, optimizer_vae_encoder, device, num_epochs, loss_tracker)


    
        # Create embeddings with frozen encoder
        # Create new models with input_dim = encoder_latent_dim for embedding models
        mlp_embedding_model = MLPRegressor(
            in_dim=encoder_latent_dim,  # Use encoder's latent dimension, not original input_dim
            hidden_size=mlp_model_cfg.get('hidden_size', [64, 32]),
            dropout=mlp_model_cfg.get('dropout', 0.3),
            batchnorm=mlp_model_cfg.get('batchnorm', True),
            activation=mlp_model_cfg.get('activation', 'ReLU')
        ).to(device)
        
        tabm_embedding_model = TabM(
            in_dim=encoder_latent_dim,  # Use encoder's latent dimension, not original input_dim
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

      

        mlp_embedding = EncoderEmbedding(model_encoder, mlp_embedding_model,freeze_encoder=True)
        tabm_embedding = EncoderEmbedding(model_encoder, tabm_embedding_model, freeze_encoder=True)


        mlp_vae_embedding = VAEEncoderMLP(model_vae_encoder, mlp_vae_embedding_model, use_mu=True, use_log_var=False, freeze_encoder=True)
        tabm_vae_embedding = VAEEncoderMLP(model_vae_encoder, tabm_vae_embedding_model, use_mu=True, use_log_var=False, freeze_encoder=True)

        optimizer_mlp_embedding = Adam(mlp_embedding.parameters(), lr=lr)
        optimizer_tabm_embedding = Adam(tabm_embedding.parameters(), lr=lr)

        optimizer_mlp_vae_embedding = Adam(mlp_vae_embedding.parameters(), lr=lr)
        optimizer_tabm_vae_embedding = Adam(tabm_vae_embedding.parameters(), lr=lr)


        

        # train the MLP and TabM
        print(f'\nStarting training...')
        print(f'Epoch | MLP train loss | MLP val loss | TabM train loss | TabM val loss | MLP embedding train loss | MLP embedding val loss | TabM embedding train loss | TabM embedding val loss | MLP VAE embedding train loss | MLP VAE embedding val loss | TabM VAE embedding train loss | TabM VAE embedding val loss')
        print('-' * 150)
        
        for epoch in range(num_epochs):
            train_loss_mlp = train_loop(model_mlp, train_loader, optimizer_mlp, loss_fn, device, model_type='mlp')
            train_loss_tabm = train_loop(model_tabm, train_loader, optimizer_tabm, loss_fn, device, model_type='tabm')
            test_loss_mlp, test_r2_mlp = eval_loop(model_mlp, test_loader, loss_fn, device, model_type='mlp')
            test_loss_tabm, test_r2_tabm = eval_loop(model_tabm, test_loader, loss_fn, device, model_type='tabm')
            train_loss_mlp_embedding = train_loop(mlp_embedding, train_loader, optimizer_mlp_embedding, loss_fn, device, model_type='mlp')
            test_loss_mlp_embedding, test_r2_mlp_embedding = eval_loop(mlp_embedding, test_loader, loss_fn, device, model_type='mlp')
            train_loss_tabm_embedding = train_loop(tabm_embedding, train_loader, optimizer_tabm_embedding, loss_fn, device, model_type='tabm')
            test_loss_tabm_embedding, test_r2_tabm_embedding = eval_loop(tabm_embedding, test_loader, loss_fn, device, model_type='tabm')
            train_loss_mlp_vae_embedding = train_loop(mlp_vae_embedding, train_loader, optimizer_mlp_vae_embedding, loss_fn, device, model_type='mlp')
            test_loss_mlp_vae_embedding, test_r2_mlp_vae_embedding = eval_loop(mlp_vae_embedding, test_loader, loss_fn, device, model_type='mlp')
            train_loss_tabm_vae_embedding = train_loop(tabm_vae_embedding, train_loader, optimizer_tabm_vae_embedding, loss_fn, device, model_type='tabm')
            test_loss_tabm_vae_embedding, test_r2_tabm_vae_embedding = eval_loop(tabm_vae_embedding, test_loader, loss_fn, device, model_type='tabm')

            # update the loss tracker for each model
            loss_tracker.update(epoch, 'mlp', train_loss_mlp, test_loss_mlp, test_r2_mlp)
            loss_tracker.update(epoch, 'tabm', train_loss_tabm, test_loss_tabm, test_r2_tabm)
            loss_tracker.update(epoch, 'mlp_embedding', train_loss_mlp_embedding, test_loss_mlp_embedding, test_r2_mlp_embedding)
            loss_tracker.update(epoch, 'tabm_embedding', train_loss_tabm_embedding, test_loss_tabm_embedding, test_r2_tabm_embedding)
            loss_tracker.update(epoch, 'mlp_vae_embedding', train_loss_mlp_vae_embedding, test_loss_mlp_vae_embedding, test_r2_mlp_vae_embedding)
            loss_tracker.update(epoch, 'tabm_vae_embedding', train_loss_tabm_vae_embedding, test_loss_tabm_vae_embedding, test_r2_tabm_vae_embedding)
            print(f'{epoch + 1:5d}  | {train_loss_mlp:13.4f}  | {test_loss_mlp:11.4f} | {train_loss_tabm:13.4f}  | {test_loss_tabm:11.4f} | {train_loss_mlp_embedding:13.4f}  | {test_loss_mlp_embedding:11.4f} | {train_loss_tabm_embedding:13.4f}  | {test_loss_tabm_embedding:11.4f} | {train_loss_mlp_vae_embedding:13.4f}  | {test_loss_mlp_vae_embedding:11.4f} | {train_loss_tabm_vae_embedding:13.4f}  | {test_loss_tabm_vae_embedding:11.4f}')
            
            # save the checkpoint (ensure all values are float for comparison)

            is_best_mlp = (test_loss_mlp == float(loss_tracker.best_val_loss.get('mlp', float('inf'))))
            is_best_tabm = (test_loss_tabm == float(loss_tracker.best_val_loss.get('tabm', float('inf'))))
            is_best_mlp_embedding = (test_loss_mlp_embedding == float(loss_tracker.best_val_loss.get('mlp_embedding', float('inf'))))
            is_best_tabm_embedding = (test_loss_tabm_embedding == float(loss_tracker.best_val_loss.get('tabm_embedding', float('inf'))))
            is_best_mlp_vae_embedding = (test_loss_mlp_vae_embedding == float(loss_tracker.best_val_loss.get('mlp_vae_embedding', float('inf'))))
            is_best_tabm_vae_embedding = (test_loss_tabm_vae_embedding == float(loss_tracker.best_val_loss.get('tabm_vae_embedding', float('inf'))))
            save_every_n_epochs = 10
            should_save_checkpoint = ((epoch + 1) % save_every_n_epochs == 0) or (epoch + 1 == num_epochs)

            if should_save_checkpoint or is_best_mlp:
                save_checkpoint(
                    experiment_dir, epoch, model_mlp, optimizer_mlp,
                    train_loss_mlp, test_loss_mlp, 'mlp', is_best_mlp,
                    save_regular_checkpoint=should_save_checkpoint
                )
            
            
            if should_save_checkpoint or is_best_tabm:
                save_checkpoint(
                    experiment_dir, epoch, model_tabm, optimizer_tabm,
                    train_loss_tabm, test_loss_tabm, 'tabm', is_best_tabm,
                    save_regular_checkpoint=should_save_checkpoint
                )
            
            if should_save_checkpoint or is_best_mlp_embedding:
                save_checkpoint(
                    experiment_dir, epoch, mlp_embedding, optimizer_mlp_embedding,
                    train_loss_mlp_embedding, test_loss_mlp_embedding, 'mlp_embedding', is_best_mlp_embedding,
                    save_regular_checkpoint=should_save_checkpoint
                )
            
            if should_save_checkpoint or is_best_tabm_embedding:
                save_checkpoint(
                    experiment_dir, epoch, tabm_embedding, optimizer_tabm_embedding,
                    train_loss_tabm_embedding, test_loss_tabm_embedding, 'tabm_embedding', is_best_tabm_embedding,
                    save_regular_checkpoint=should_save_checkpoint
                )
            
            if should_save_checkpoint or is_best_mlp_vae_embedding:
                save_checkpoint(
                    experiment_dir, epoch, mlp_vae_embedding, optimizer_mlp_vae_embedding,
                    train_loss_mlp_vae_embedding, test_loss_mlp_vae_embedding, 'mlp_vae_embedding', is_best_mlp_vae_embedding,
                    save_regular_checkpoint=should_save_checkpoint
                )
            
            if should_save_checkpoint or is_best_tabm_vae_embedding:
                save_checkpoint(
                    experiment_dir, epoch, tabm_vae_embedding, optimizer_tabm_vae_embedding,
                    train_loss_tabm_vae_embedding, test_loss_tabm_vae_embedding, 'tabm_vae_embedding', is_best_tabm_vae_embedding,
                    save_regular_checkpoint=should_save_checkpoint
                )
        
            
        
        print(f'\nTraining complete!')
        
        # save the plots and the losses
        loss_tracker.save_plots(experiment_dir)
        loss_tracker.save_losses_to_file(experiment_dir)
        
        print(f'\nAll results have been saved to: {experiment_dir}')
        print(f'  - config file: {experiment_dir / "configs"}')
        print(f'  - checkpoint: {experiment_dir / "checkpoints"}')
        print(f'  - loss plots: {experiment_dir / "plots"}')
        print(f'  - loss records: {experiment_dir / "losses.txt"}')
        
        return experiment_dir
    
    except Exception as e:
        print(f'Error: {e}')
        raise

