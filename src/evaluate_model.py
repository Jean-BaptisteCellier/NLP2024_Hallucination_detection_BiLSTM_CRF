import torch
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def evaluate_model_bilstm_crf2(model, data_loader):
    model.eval()
    all_predictions = []
    true_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for x_tokens, x_additional_features, labels in data_loader:
            # Forward pass through model
            outputs = model(x_tokens, x_additional_features, labels)
            
            if isinstance(outputs, tuple):
                emissions, loss = outputs
            else:
                emissions = outputs
                loss = None

            # Adjust mask to match emissions shape
            seq_len = emissions.size(1)
            mask = (x_tokens[:, :seq_len] != model.padding_idx)

            # Decode using CRF
            batch_predictions = model.crf.decode(emissions, mask=mask)

            # Collect true and predicted labels
            for preds, true_seq, mask_seq in zip(batch_predictions, labels, mask):
                valid_len = mask_seq.sum().item()
                all_predictions.extend(preds[:valid_len])
                true_labels.extend(true_seq[:valid_len].numpy())

            if loss is not None:
                total_loss += loss.item()

    avg_loss = total_loss / len(data_loader) if total_loss > 0 else None

    accuracy = accuracy_score(true_labels, all_predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(true_labels, all_predictions))

    return avg_loss, all_predictions, np.stack(true_labels).tolist()


