import torch
from torch.nn import MSELoss
from src.training.vae_loss import vae_loss


def train_encoder(model_encoder, train_loader, optimizer_encoder, loss_fn, device, num_epochs, loss_tracker):
    model_encoder.train()
    
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
        loss_tracker.update(epoch, 'encoder', avg_train_loss, avg_train_loss)
    
    # Set encoder to eval mode after training to ensure consistent behavior in embeddings
    model_encoder.eval()
    print(f'\nEncoder training complete! Loss: {avg_train_loss:.4f}')
    return avg_train_loss


def train_vae_encoder(model_vae_encoder, train_loader, optimizer_vae_encoder, device, num_epochs, loss_tracker):
    model_vae_encoder.train()
    
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
        loss_tracker.update(epoch, 'vae_encoder', avg_train_loss, avg_train_loss)
    
    # Set VAE encoder to eval mode after training to ensure consistent behavior in embeddings
    model_vae_encoder.eval()
    print(f'\nVAEEncoderMLP training complete! Loss: {avg_train_loss:.4f}')
    return avg_train_loss