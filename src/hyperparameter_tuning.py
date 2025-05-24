from sklearn.metrics import f1_score
import torch
import model_BiLSTMCRF as mod
import data_prep as prep
import train_val_model as tv

def objective(trial, vocab_size, output_dim, padding_idx, additional_feature_dim, train_data,
              val_data, test_data):
    hidden_dim = trial.suggest_int("hidden_dim", 128, 512, step=64)
    embedding_dim = trial.suggest_int("embedding_dim", 100, 300, step=50)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    epochs = 10
    batch_size = trial.suggest_categorical("batch_size", [8,16,32])

    model = mod.BiLSTM_CRF_Model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        padding_idx=padding_idx,
        additional_feature_dim=additional_feature_dim
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, val_loader, _ = prep.create_data_loaders(train_data, val_data, test_data, batch_size=batch_size)

    result = tv.train_model_bilstm_crf(
        model, train_loader, val_loader, optimizer, epochs=epochs
    )
    print(result)
    _, _, val_predictions, val_true_labels = tv.train_model_bilstm_crf(
        model, train_loader, val_loader, optimizer, epochs=epochs
    )

    val_f1_score = f1_score(val_true_labels, val_predictions, average="weighted")
    return val_f1_score 
