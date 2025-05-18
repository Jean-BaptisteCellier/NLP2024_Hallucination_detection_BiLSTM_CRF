import torch

def train_model_bilstm_crf(model, train_loader, val_loader, optimizer, epochs=10):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train() 
        total_loss = 0.0

        # Training loop
        for batch_idx, batch in enumerate(train_loader):
            x_tokens, x_additional_features, labels = batch

            # Forward pass
            _, loss = model(x_tokens, x_additional_features, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        val_predictions = []
        val_true_labels = []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(x_tokens, x_additional_features, labels)
                if isinstance(outputs, tuple):
                    emissions, loss = outputs
                else:
                    emissions, loss = outputs, None

                total_val_loss += loss.item()
                predictions = model.crf.decode(emissions)

                # Collect predictions and true labels
                for pred, label, mask in zip(predictions, labels, (x_tokens != model.padding_idx)):
                    val_predictions.extend(pred[:mask.sum()])
                    val_true_labels.extend(label[:mask.sum()])

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses, val_predictions, val_true_labels