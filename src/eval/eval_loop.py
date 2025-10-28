import torch

def eval_loop(model, loader, loss_fn, device, model_type: str):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, labels in loader:
            x = x.to(device)
            labels = labels.to(device)
            
            #forward pass
            outputs_raw = model(x)

            if model_type == 'mlp':
                outputs = outputs_raw
                loss = loss_fn(outputs, labels)
            elif model_type =='tabm':
                outputs_mean = outputs_raw.mean(dim=1)
                loss = loss_fn(outputs_mean, labels)

            else:
                raise ValueError(f'Unknow model type {model_type}')
            

            
            total_loss += loss.item()

    return total_loss / len(loader)

