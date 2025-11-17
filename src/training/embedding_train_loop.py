import torch
import math
from torch.nn import MSELoss
from pathlib import Path
from src.training.vae_loss import vae_loss
from src.utils.save_utils import save_checkpoint


def eval_encoder(model_encoder, val_loader, loss_fn, device):
    model_encoder.eval()
    total_loss = 0.0
    valid_batches = 0
    
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            z, recon = model_encoder(x)
            loss = loss_fn(recon, x)
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                valid_batches += 1
    
    if valid_batches == 0:
        return float('nan')
    return total_loss / valid_batches


def train_encoder(model_encoder, train_loader, val_loader, optimizer_encoder, loss_fn, device, num_epochs, loss_tracker, experiment_dir=None):
    model_encoder.train()
    best_val_loss = float('inf')
    best_epoch = -1
    
    for epoch in range(num_epochs):
        epoch_loss_encoder = 0.0
        for x, _ in train_loader:
            x = x.to(device)
            z, recon = model_encoder(x)
            loss_encoder = loss_fn(recon, x)
            optimizer_encoder.zero_grad()
            loss_encoder.backward()
            optimizer_encoder.step()
            epoch_loss_encoder += loss_encoder.item()
        avg_train_loss = epoch_loss_encoder / len(train_loader)
        
        avg_val_loss = eval_encoder(model_encoder, val_loader, loss_fn, device)
        loss_tracker.update(epoch, 'encoder', avg_train_loss, avg_val_loss)
        
        if not (math.isnan(avg_val_loss) or math.isinf(avg_val_loss)):
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                if experiment_dir is not None:
                    is_best = True
                    save_checkpoint(
                        experiment_dir, epoch, model_encoder, optimizer_encoder,
                        avg_train_loss, avg_val_loss, 'encoder', is_best,
                        save_regular_checkpoint=False
                    )
        
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    print(f'\nEncoder training complete! Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch+1}')
    return best_val_loss


def eval_vae_encoder(model_vae_encoder, val_loader, device):
    model_vae_encoder.eval()
    total_loss = 0.0
    valid_batches = 0
    
    with torch.no_grad():
        for x, _ in val_loader:
            x = x.to(device)
            recon, mu, log_var, z = model_vae_encoder(x, training=False)
            loss, _ = vae_loss(x, recon, mu, log_var, beta=1.0)
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                valid_batches += 1
    
    if valid_batches == 0:
        return float('nan')
    return total_loss / valid_batches


def train_vae_encoder(model_vae_encoder, train_loader, val_loader, optimizer_vae_encoder, device, num_epochs, loss_tracker, experiment_dir=None):
    model_vae_encoder.train()
    best_val_loss = float('inf')
    best_epoch = -1
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        valid_batches = 0
        beta = min(1.0, epoch / 20)  # Beta-VAE: gradually increase KL weight
        for x, _ in train_loader:
            x = x.to(device)
            recon, mu, log_var, z = model_vae_encoder(x, training=True)
            loss, _ = vae_loss(x, recon, mu, log_var, beta)
            
            # Check for NaN/Inf and skip if found
            if torch.isnan(loss) or torch.isinf(loss):
                print(f'Warning: NaN/Inf loss detected at epoch {epoch}, skipping batch')
                continue
            
            optimizer_vae_encoder.zero_grad()
            loss.backward()
            optimizer_vae_encoder.step()
            epoch_loss += loss.item()
            valid_batches += 1
        
        if valid_batches == 0:
            avg_train_loss = float('nan')
        else:
            avg_train_loss = epoch_loss / valid_batches
        
        avg_val_loss = eval_vae_encoder(model_vae_encoder, val_loader, device)
        loss_tracker.update(epoch, 'vae_encoder', avg_train_loss, avg_val_loss)
        
        if not (math.isnan(avg_val_loss) or math.isinf(avg_val_loss)):
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch
                if experiment_dir is not None:
                    is_best = True
                    save_checkpoint(
                        experiment_dir, epoch, model_vae_encoder, optimizer_vae_encoder,
                        avg_train_loss, avg_val_loss, 'vae_encoder', is_best,
                        save_regular_checkpoint=False
                    )
        
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    print(f'\nVAEEncoderMLP training complete! Best Val Loss: {best_val_loss:.4f} at epoch {best_epoch+1}')
    return best_val_loss