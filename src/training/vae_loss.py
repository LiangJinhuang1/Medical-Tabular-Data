import torch
import torch.nn.functional as F

def vae_loss(x, recon_x, mu, log_std, beta=1.0):
    mse = F.mse_loss(recon_x, x)
    # Use numerically stable computation: kl = 0.5 * sum(exp(2*log_std) + mu^2 - 1 - 2*log_std)
    log_std_clamped = torch.clamp(log_std, min=-10, max=10)
    kl = -0.5 * torch.mean(1 + 2*log_std_clamped - mu.pow(2) - (2*log_std_clamped).exp())
    return mse + beta * kl, {'mse': mse.item(), 'kl': kl.item()}
