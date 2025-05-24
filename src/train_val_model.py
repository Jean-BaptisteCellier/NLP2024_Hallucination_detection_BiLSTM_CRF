import torch
import model_BiLSTMCRF as mod

def train_model_bilstm_crf(model, train_loader, val_loader, optimizer, epochs=10):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x_tokens, x_additional_features, labels in train_loader:
            
            _, loss = model(x_tokens, x_additional_features, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        total_val_loss = 0.0
        val_predictions = []
        val_true_labels = []

        with torch.no_grad():
            for x_tokens, x_additional_features, labels in val_loader:
                x_tokens, x_additional_features, labels = x_tokens, x_additional_features, labels
                emissions, loss = model(x_tokens, x_additional_features, labels)

                total_val_loss += loss.item()
                mask = x_tokens[:, :emissions.size(1)] != model.padding_idx
                predictions = model.crf.decode(emissions, mask=mask)

                for pred_seq, label_seq, mask_seq in zip(predictions, labels, mask):
                    valid_len = mask_seq.sum().item()
                    val_predictions.extend(pred_seq[:valid_len])
                    val_true_labels.extend(label_seq[:valid_len].tolist())

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses, val_predictions, val_true_labels, best_model