def train_loop(model, loader, optimizer, loss_fn, device, model_type: str):
    model.train()
    total_loss = 0

    for x, labels in loader:
        x = x.to(device)
        labels = labels.to(device)
        
        #forward pass
        outputs_raw = model(x)

        if model_type == 'mlp':
            loss = loss_fn(outputs_raw,labels)
        elif model_type == 'tabm':
            # TabM output shape: (batch_size, k, d_out)
            #outputs = outputs.mean(dim=1)
            per_head_losses = []
            K = outputs_raw.shape[1]
            for k in range(K):
                head_pred = outputs_raw[:,k,:]
                loss_k = loss_fn(head_pred, labels)
                per_head_losses.append(loss_k)
            loss = sum(per_head_losses)/len(per_head_losses)
        
        else:
            raise ValueError(f'Unknown model type {model_type}')


        #backward pass
        optimizer.zero_grad()
        loss.backward()

        #update parameters
        optimizer.step()

        total_loss += loss.item()
    return total_loss/len(loader)

    