import torch
import torch.nn.functional as F 
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model_bilstm_crf(model, data_loader):
    all_predictions = [] 
    true_labels = [] 
    hard_labels = []
    soft_labels = []
    total_loss = 0

    with torch.no_grad():
        for x_tokens, x_additional_features, labels in data_loader:
            mask = (x_tokens != model.padding_idx)
            outputs = model(x_tokens, x_additional_features, labels)
            
            if isinstance(outputs, tuple):
                emissions, loss = outputs 
            else:
                emissions = outputs
                loss = None

            # Soft and hard labels
            probabilities = F.softmax(emissions, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)

            max_len = probabilities.size(1)
            x_tokens = x_tokens[:, :max_len]
            labels = labels[:, :max_len]
            mask = (x_tokens != model.padding_idx)

            valid_predictions = predictions[mask]
            valid_labels = labels[mask]
            valid_probs = probabilities[mask]

            # Convert to lists and extend results
            all_predictions.extend(valid_predictions.cpu().numpy().tolist())
            true_labels.extend(valid_labels.cpu().numpy().tolist())
            hard_labels.extend(valid_predictions.cpu().tolist())
            soft_labels.extend(valid_probs.cpu().tolist())

            if loss is not None:
                print("Loss:", loss.item())
                total_loss += loss.item()

    avg_loss = total_loss / len(data_loader) if total_loss > 0 else None

    accuracy = accuracy_score(true_labels, all_predictions)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(true_labels, all_predictions))

    return avg_loss, all_predictions, true_labels, hard_labels, soft_labels
