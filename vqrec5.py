import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SinkhornTransformer


class VQReq(nn.Module):
    def __init__(self, code_dim, code_cap, codebook_size, hidden_size, num_layers, num_heads, dropout, temperature,
                 loss_type='CE', index_assignment_flag=True, fake_idx_ratio=0.1):
        super(VQReq, self).__init__()
        self.code_dim = code_dim
        self.code_cap = code_cap
        self.codebook_size = codebook_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.temperature = temperature
        self.loss_type = loss_type
        self.index_assignment_flag = index_assignment_flag
        self.fake_idx_ratio = fake_idx_ratio

        self.ITEM_SEQ = 'item_seq'
        self.ITEM_SEQ_LEN = 'item_seq_len'
        self.POS_ITEM_ID = 'pos_item_id'

        self.pq_codes = nn.Parameter(torch.Tensor(codebook_size, code_dim))
        nn.init.xavier_uniform_(self.pq_codes)

        self.position_embedding = nn.Embedding(code_cap, hidden_size)

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.trm_encoder = SinkhornTransformer(d_model=hidden_size, nhead=num_heads, num_layers=num_layers,
                                                dim_feedforward=hidden_size*4, dropout=dropout,
                                                activation='gelu')

        if index_assignment_flag:
            self.reassigned_code_embedding = nn.Parameter(torch.Tensor(codebook_size, hidden_size))
            nn.init.xavier_uniform_(self.reassigned_code_embedding)
        else:
            self.pq_code_embedding = nn.Embedding(codebook_size * code_cap, hidden_size)

        if temperature > 0:
            self.loss_fct = nn.CrossEntropyLoss(reduction='mean')
        else:
            self.loss_fct = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        pq_code_seq = self.pq_codes[item_seq]
        if self.index_assignment_flag:
            pq_code_emb = F.embedding(pq_code_seq, self.reassigned_code_embedding, padding_idx=0).mean(dim=-2)
        else:
            pq_code_emb = self.pq_code_embedding(pq_code_seq).mean(dim=-2)
        input_emb = pq_code_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        trm_output = self.trm_encoder(input_emb, None, item_seq_len)
        output = trm_output[0]
        output = self.gather_indexes(output, item_seq_len - 1)
        return output  # [B H]

    def calculate_item_emb(self):
        if self.index_assignment_flag:
            pq_code_emb = F.embedding(self.pq_codes, self.reassigned_code_embedding, padding_idx=0).mean(dim=-2)
        else:
            pq_code_emb = self.pq_code_embedding(self.pq_codes).mean(dim=-2)
        return pq_code_emb  # [B H]

    def generate_fake_neg_item_emb(self, item_index):
        rand_idx = torch.randint_like(input=item_index, high=self.code_cap)
        # flatten pq codes
        base_id = (torch.arange(self.code_dim).to(item
