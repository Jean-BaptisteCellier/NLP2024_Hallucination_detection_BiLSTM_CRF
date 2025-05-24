import torch
import torch.nn as nn
from torchcrf import CRF

class BiLSTM_CRF_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, padding_idx, additional_feature_dim):
        super(BiLSTM_CRF_Model, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        # BiLSTM layer
        self.lstm = nn.LSTM(
            embedding_dim + additional_feature_dim, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=True
        )
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 [bidirectional]
        # CRF layer
        self.crf = CRF(output_dim, batch_first=True)
        self.padding_idx = padding_idx

    def forward(self, x_tokens, x_additional_features, labels=None):
        # Create a mask for non-padding tokens
        mask = x_tokens != self.padding_idx
        # Embedding
        embedded_tokens = self.embedding(x_tokens)
        # Apply mask to embedded tokens and additional features
        embedded_tokens = embedded_tokens * mask.unsqueeze(-1).type(embedded_tokens.dtype)
        additional_features = x_additional_features * mask.unsqueeze(-1).type(x_additional_features.dtype)
        # Combine embeddings and additional features
        combined_input = torch.cat((embedded_tokens, additional_features), dim=-1)
        # Pack padded sequence
        seq_lengths = mask.sum(dim=1).cpu()
        packed_input = nn.utils.rnn.pack_padded_sequence(combined_input, seq_lengths, batch_first=True, enforce_sorted=False)
        # Forward through BiLSTM
        packed_output, _ = self.lstm(packed_input)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True) #padding_value=self.padding_idx
        # Fully connected layer
        emissions = self.fc(lstm_output)
        if torch.isnan(emissions).any():
            print("NaN in emissions!")
        if labels is not None:
            seq_len = emissions.size(1)
            labels_clean = labels[:, :seq_len].clone()
            mask_clean = mask[:, :seq_len]
            labels_clean[~mask_clean] = self.padding_idx
            # mask_clean = (x_tokens[:, :seq_len] != self.padding_idx)
            loss = -self.crf(emissions, labels_clean, mask=mask_clean)
            return emissions, loss
        return emissions