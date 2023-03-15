import torch
import torch.nn as nn
import torch.nn.functional as F


class VQRec(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, dropout_rate):
        super(VQRec, self).__init__()

        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=0)
        self.gru = nn.GRU(emb_size, hidden_size, num_layers, dropout=dropout_rate, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, vocab_size)

        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

    def forward(self, input_ids, input_lens, h_n=None):
        input_embs = self.embedding(input_ids)
        packed_input = nn.utils.rnn.pack_padded_sequence(input_embs, input_lens, batch_first=True, enforce_sorted=False)

        packed_output, h_n = self.gru(packed_input, h_n)

        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        output = F.relu(self.fc1(output))
        logits = self.fc2(output)

        return logits, h_n

    def calculate_loss(self, interaction):
        input_ids = interaction["input_ids"]
        target_ids = interaction["target_ids"]
        input_lens = interaction["input_lens"]

        logits, _ = self.forward(input_ids, input_lens)

        loss = F.cross_entropy(logits.view(-1, self.vocab_size), target_ids.view(-1))

        return loss
