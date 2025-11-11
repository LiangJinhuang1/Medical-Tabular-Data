import torch
from sklearn.metrics import r2_score
import numpy as np

def eval_loop(model, loader, loss_fn, device, model_type: str):
    model.eval()
    total_loss = 0
    valid_batches = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)
            labels = labels.to(device)
            
            if labels.dim() == 1:
                labels = labels.unsqueeze(1) 
            
            #forward pass
            outputs_raw = model(x)

            if model_type == 'mlp':
                outputs = outputs_raw
            elif model_type =='tabm':
                outputs_mean = outputs_raw.mean(dim=1)
                outputs = outputs_mean
            else:
                raise ValueError(f'Unknown model type {model_type}')
            
            
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(1)
            
            loss = loss_fn(outputs, labels)


            all_predictions.append(outputs)
            all_labels.append(labels)
            
            total_loss += loss.item()
            valid_batches += 1

    
    if valid_batches == 0:
        avg_loss = float('nan')
    else:
        avg_loss = total_loss / valid_batches
    

    if len(all_predictions) > 0:
        all_predictions = torch.cat(all_predictions, dim=0).cpu().numpy()
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
        try:
            r2 = r2_score(all_labels, all_predictions)
        except ValueError as e:
            print(f'Error calculating R2 score: {e}')
            print(f'Labels shape: {all_labels.shape}')
            print(f'Predictions shape: {all_predictions.shape}')
            r2 = float('-inf')
    else:
        r2 = float('-inf')
    
    return avg_loss, r2

