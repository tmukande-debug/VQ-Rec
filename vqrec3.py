import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder

from sinkhorn_transformer import SinkhornTransformer

class VQRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        # VQRec args
        self.code_dim = config['code_dim']
        self.code_cap = config['code_cap']
        self.pq_codes = dataset.pq_codes
        self.temperature = config['temperature']
        self.index_assignment_flag = False
        self.sinkhorn_iter = config['sinkhorn_iter']
        self.fake_idx_ratio = config['fake_idx_ratio']

        self.train_stage = config['train_stage']
        assert self.train_stage in [
            'pretrain', 'inductive_ft'
        ], f'Unknown train stage: [{self.train_stage}]'

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.initializer_range = config['initializer_range']
        self.loss_type = config['loss_type']

        # define layers and loss
        self.pq_code_embedding = nn.Embedding(
            self.code_dim * (1 + self.code_cap), self.hidden_size, padding_idx=0)
        self.reassigned_code_embedding = None

        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.trm_encoder = SinkhornTransformer(
            num_tokens=self.code_dim * (1 + self.code_cap),
            dim=self.hidden_size,
            depth=self.n_layers,
            heads=self.n_heads,
            ff_glu=True,
            dim_head=None,
            reversible=True,
            ff_dropout=self.hidden_dropout_prob,
            attn_dropout=self.attn_dropout_prob,
            loss_type='CE',
            num_patches=None,
            row_normalization=True,
            column_normalization=True,
            sinkhorn_iter=self.sinkhorn_iter,
            n_sortcut=None,
            learn_unary=None,
            hard_k=None
        )

        self.trans_matrix = nn.Parameter(torch.randn(self.code_dim, self.code_cap + 1, self.code_cap + 1))

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        if self.loss_type == 'BPR':
            raise NotImplementedError()
        elif self.loss_type == 'CE':
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['CE']!")

        # parameters initialization
        self.apply(self._init_weights)

  def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
  
  def forward(self, seq):
    # PQ coding
    seq_codes = self.pq_codes[seq].view(-1, self.code_cap)
    seq_codes = torch.cat((seq_codes, torch.zeros(seq_codes.size(0), 1).to(seq_codes.device)), dim=-1).long()
    seq_codes = seq_codes.view(-1)
    seq_codes = self.pq_code_embedding(seq_codes)
    seq_codes = seq_codes.view(-1, self.max_seq_length, self.hidden_size)

    # Sinkhorn Transformer encoding
    seq_len = seq.size(1)
    position_ids = torch.arange(seq_len, dtype=torch.long, device=seq.device)
    position_ids = position_ids.unsqueeze(0).expand_as(seq)
    position_embeds = self.position_embedding(position_ids)
    seq_codes += position_embeds

    # Reshape the seq_codes tensor from (batch_size, seq_len, hidden_size) to (batch_size * seq_len, hidden_size)
    seq_codes = seq_codes.view(-1, self.hidden_size)

    # Use the Sinkhorn Transformer to encode the sequence
    seq_codes = self.trm_encoder(seq_codes)

    # Reshape the seq_codes tensor back to (batch_size, seq_len, hidden_size)
    seq_codes = seq_codes.view(-1, seq_len, self.hidden_size)

    # Re-assign the learned codes
    if self.index_assignment_flag:
        assigned_codes, assigned_indices, num_fake_indices = self._assign_codes(seq_codes.detach())
        assigned_codes = assigned_codes.to(seq.device)
        assigned_indices = assigned_indices.to(seq.device)
        num_fake_indices = num_fake_indices.to(seq.device)

        fake_idx = torch.randint(0, assigned_indices.size(0),
                                  size=(int(num_fake_indices),),
                                  device=assigned_indices.device)
        fake_indices = torch.full((int(num_fake_indices),), -1, dtype=torch.long, device=assigned_indices.device)
        fake_codes = torch.randn((int(num_fake_indices), self.hidden_size),
                                 device=assigned_codes.device)
        assigned_indices = torch.cat([assigned_indices, fake_indices], dim=0)
        assigned_codes = torch.cat([assigned_codes, fake_codes], dim=0)
        assigned_codes = assigned_codes[torch.argsort(assigned_indices)]

        # store reassigned codes and indices
        self.reassigned_code_embedding = nn.Parameter(assigned_codes, requires_grad=False)
    else:
        assigned_codes = self.reassigned_code_embedding

    # Compute the logits and loss
    logits = torch.matmul(seq_codes, assigned_codes.t())
    logits /= self.temperature

    if self.loss_type == 'CE':
        loss = self.loss_fct(logits.view(-1, self.code_dim * (1 + self.code_cap)), seq.view(-1))
    else:
        raise NotImplementedError()

    return {'logits': logits, 'loss': loss}
