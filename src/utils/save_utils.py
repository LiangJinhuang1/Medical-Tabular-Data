import os
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import shutil


def create_experiment_dir(output_root: Path = None) -> Path:

    if output_root is None:
        project_root = Path(__file__).parent.parent.parent.resolve()
        output_root = project_root / "output"
    
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = output_root / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)
    

    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "configs").mkdir(exist_ok=True)
    (experiment_dir / "plots").mkdir(exist_ok=True)
    
    return experiment_dir


def save_configs(experiment_dir: Path, config_dict: Dict[str, Any]):

    configs_dir = experiment_dir / "configs"
    

    config_file = configs_dir / "full_config.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    


def save_checkpoint(
    experiment_dir: Path,
    epoch: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loss: float,
    val_loss: float,
    model_name: str,
    is_best: bool = False,
    save_regular_checkpoint: bool = True
):

    checkpoints_dir = experiment_dir / "checkpoints"
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model_name': model_name
    }
    
    # Save regular checkpoint only if requested
    if save_regular_checkpoint:
        checkpoint_path = checkpoints_dir / f"{model_name}_epoch_{epoch+1}.pt"
        torch.save(checkpoint, checkpoint_path)
    
    # Always save best model
    if is_best:
        best_checkpoint_path = checkpoints_dir / f"{model_name}_best.pt"
        torch.save(checkpoint, best_checkpoint_path)


class LossTracker:
    
    def __init__(self):
        self.train_losses: Dict[str, List[float]] = {}
        self.val_losses: Dict[str, List[float]] = {}
        self.val_r2_scores: Dict[str, List[float]] = {}
        self.best_val_loss: Dict[str, float] = {}
        self.best_val_r2: Dict[str, float] = {}
        self.best_epoch: Dict[str, int] = {}  # Best epoch based on loss
        self.best_r2_epoch: Dict[str, int] = {}  # Best epoch based on R2
    
    def update(self, epoch: int, model_name: str, train_loss: float, val_loss: float, val_r2: float = None):
        """
        Update losses for a specific model.
        
        Args:
            epoch: Current epoch number
            model_name: Name of the model (e.g., 'mlp', 'tabm', 'custom_model')
            train_loss: Training loss for this epoch
            val_loss: Validation loss for this epoch
            val_r2: Validation R2 score for this epoch (optional)
        """
        # Initialize lists for new models
        if model_name not in self.train_losses:
            self.train_losses[model_name] = []
            self.val_losses[model_name] = []
            self.val_r2_scores[model_name] = []
            self.best_val_loss[model_name] = float('inf')
            self.best_val_r2[model_name] = float('-inf')
            self.best_epoch[model_name] = -1
            self.best_r2_epoch[model_name] = -1
        
        # Append losses 
        train_loss = float(train_loss)
        val_loss = float(val_loss)
            
        self.train_losses[model_name].append(train_loss)
        self.val_losses[model_name].append(val_loss)
        
        # Append R2 score if provided
        if val_r2 is not None:
            val_r2 = float(val_r2)
            self.val_r2_scores[model_name].append(val_r2)
        else:
            self.val_r2_scores[model_name].append(float('nan'))
        
        # Update best model
        best_val = self.best_val_loss[model_name]
        if not isinstance(best_val, (int, float)):
            current_best = float('inf')
            self.best_val_loss[model_name] = current_best
        else:
            current_best = float(best_val)
            
        if val_loss < current_best:
            self.best_val_loss[model_name] = val_loss
            self.best_epoch[model_name] = epoch
        
        # Update best R2
        if val_r2 is not None and not np.isnan(val_r2):
            current_best_r2 = self.best_val_r2[model_name]
            if not isinstance(current_best_r2, (int, float)) or np.isnan(current_best_r2):
                current_best_r2 = float('-inf')
            else:
                current_best_r2 = float(current_best_r2)
            
            if val_r2 > current_best_r2:
                self.best_val_r2[model_name] = val_r2
                self.best_r2_epoch[model_name] = epoch
    
    def save_plots(self, experiment_dir: Path, figsize_per_model: tuple = (7, 5)):
        if not self.train_losses:
            print("No loss data to plot.")
            return
        
        plots_dir = experiment_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        model_names = sorted(self.train_losses.keys())
        num_models = len(model_names)
        
        # Calculate grid layout
        cols = int(num_models ** 0.5) + (1 if num_models ** 0.5 % 1 != 0 else 0)
        rows = (num_models + cols - 1) // cols
        fig_width = figsize_per_model[0] * cols
        fig_height = figsize_per_model[1] * rows
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        
        if num_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, list) else [axes]
        else:
            axes = axes.flatten()
        
        # Plot for each model
        for idx, model_name in enumerate(model_names):
            ax = axes[idx]
            train_losses = self.train_losses[model_name]
            val_losses = self.val_losses[model_name]
            epochs = range(1, len(train_losses) + 1)
            
            # Plot losses
            ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
            ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
            
            # Mark best epoch
            if self.best_epoch[model_name] >= 0:
                ax.axvline(x=self.best_epoch[model_name] + 1, color='g', linestyle='--', 
                          label=f'Best Epoch ({self.best_epoch[model_name] + 1})', alpha=0.7)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(f'{model_name.upper()} Model Loss', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        for idx in range(num_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = plots_dir / "loss_curves.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nThe loss plots have been saved to: {plot_path}")
    
    def save_losses_to_file(self, experiment_dir: Path):

        if not self.train_losses:
            print("No loss data to save.")
            return
        
        losses_file = experiment_dir / "losses.txt"
        model_names = sorted(self.train_losses.keys())
        
        with open(losses_file, 'w', encoding='utf-8') as f:
            header_parts = ["Epoch"]
            for model_name in model_names:
                header_parts.append(f"{model_name.upper()} Train")
                header_parts.append(f"{model_name.upper()} Val")
                header_parts.append(f"{model_name.upper()} R2")
            f.write(" | ".join(header_parts) + "\n")
            f.write("-" * (len(" | ".join(header_parts)) + 10) + "\n")
            
            # Find the maximum number of epochs across all models
            max_epochs = max(len(self.train_losses[model_name]) for model_name in model_names)
            for epoch in range(max_epochs):
                row_parts = [f"{epoch + 1:5d}"]
                for model_name in model_names:
                    if epoch < len(self.train_losses[model_name]):
                        train_loss = self.train_losses[model_name][epoch]
                        val_loss = self.val_losses[model_name][epoch]
                        val_r2 = self.val_r2_scores[model_name][epoch] if epoch < len(self.val_r2_scores[model_name]) else float('nan')
                    else:
                        train_loss = float('nan')
                        val_loss = float('nan')
                        val_r2 = float('nan')
                    row_parts.append(f"{train_loss:13.4f}")
                    row_parts.append(f"{val_loss:11.4f}")
                    row_parts.append(f"{val_r2:11.4f}")
                f.write(" | ".join(row_parts) + "\n")
            
            # summary
            f.write("\n" + "=" * (len(" | ".join(header_parts)) + 10) + "\n")
            for model_name in model_names:
                best_epoch = int(self.best_epoch[model_name])
                best_loss = float(self.best_val_loss[model_name])
                best_r2 = self.best_val_r2.get(model_name, float('nan'))
                best_r2_epoch = int(self.best_r2_epoch.get(model_name, -1))
                
                # best model based on loss
                if not np.isnan(best_r2) and best_r2 != float('-inf'):
                    f.write(f"Best {model_name.upper()} Model (by Loss) - Epoch: {best_epoch + 1}, Val Loss: {best_loss:.4f}, Val R2: {best_r2:.4f}\n")
                else:
                    f.write(f"Best {model_name.upper()} Model (by Loss) - Epoch: {best_epoch + 1}, Val Loss: {best_loss:.4f}\n")
                
                # best model based on R2
                if best_r2_epoch >= 0 and not np.isnan(best_r2) and best_r2 != float('-inf'):
                    best_r2_val = float(best_r2)
                    if best_r2_epoch != best_epoch:
                        # the loss at the best R2 epoch
                        best_r2_loss = self.val_losses[model_name][best_r2_epoch] if best_r2_epoch < len(self.val_losses[model_name]) else float('nan')
                        f.write(f"Best {model_name.upper()} Model (by R2) - Epoch: {best_r2_epoch + 1}, Val Loss: {best_r2_loss:.4f}, Val R2: {best_r2_val:.4f}\n")
                    else:
                        f.write(f"Best {model_name.upper()} Model (by R2) - Epoch: {best_r2_epoch + 1}, Val R2: {best_r2_val:.4f} (same as best loss epoch)\n")
        
        print(f"\nThe loss records have been saved to: {losses_file}")