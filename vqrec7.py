import torch
import torch.nn as nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder

from sinkhorn_transformer import SinkhornTransformer

def log(t, eps = 1e-6):
    return torch.log(t + eps)


def sample_gumbel(shape, device, dtype, eps=1e-6):
    u = torch.empty(shape, device=device, dtype=dtype).uniform_(0, 1)
    return -log(-log(u, eps), eps)


def sinkhorn_sorting_operator(r, n_iters=8):
    n = r.shape[1]
    for _ in range(n_iters):
        r = r - torch.logsumexp(r, dim=2, keepdim=True)
        r = r - torch.logsumexp(r, dim=1, keepdim=True)
    return torch.exp(r)


def gumbel_sinkhorn(r, n_iters=8, temperature=0.7):
    r = log(r)
    gumbel = sample_gumbel(r.shape, r.device, r.dtype)
    r = (r + gumbel) / temperature
    return sinkhorn_sorting_operator(r, n_iters)


def differentiable_topk(x, k, temperature=1.):
    *_, n, dim = x.shape
    topk_tensors = []

    for i in range(k):
        is_last = i == (k - 1)
        values, indices = (x / temperature).softmax(dim=-1).topk(1, dim=-1)
        topks = torch.zeros_like(x).scatter_(-1, indices, values)
        topk_tensors.append(topks)
        if not is_last:
            x.scatter_(-1, indices, float('-inf'))

    topks = torch.cat(topk_tensors, dim=-1)
    return topks.reshape(*_, k * n, dim)

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
            #num_tokens=self.code_dim * (1 + self.code_cap),
            dim=self.hidden_size,
            depth=self.n_layers,
            heads=self.n_heads,
            ff_glu=True,
            dim_head=None,
            reversible=True,
            ff_dropout=self.hidden_dropout_prob,
            attn_dropout=self.attn_dropout_prob,
            sinkhorn_iter=self.sinkhorn_iter,
            n_sortcut=None
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
    
    def code_projection(self):
        doubly_stochastic_matrix = gumbel_sinkhorn(torch.exp(self.trans_matrix), n_iters=self.sinkhorn_iter)
        trans = differentiable_topk(doubly_stochastic_matrix.reshape(-1, self.code_cap + 1), 1)
        trans = torch.ceil(trans.reshape(-1, self.code_cap + 1, self.code_cap + 1))
        raw_embed = self.pq_code_embedding.weight.reshape(self.code_dim, self.code_cap + 1, -1)
        trans_embed = torch.bmm(trans, raw_embed).reshape(-1, self.hidden_size)
        return trans_embed
            
    def forward(self, seq, item_seq):
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

        extended_attention_mask = self.get_attention_mask(item_seq)

    def calculate_item_emb(self):
        if self.index_assignment_flag:
            pq_code_emb = F.embedding(self.pq_codes, self.reassigned_code_embedding, padding_idx=0).mean(dim=-2)
        else:
            pq_code_emb = self.pq_code_embedding(self.pq_codes).mean(dim=-2)
        return pq_code_emb  # [B H]

    def generate_fake_neg_item_emb(self, item_index):
        rand_idx = torch.randint_like(input=item_index, high=self.code_cap)
        # flatten pq codes
        base_id = (torch.arange(self.code_dim).to(item_index.device) * (self.code_cap + 1)).unsqueeze(0)
        rand_idx = rand_idx + base_id + 1
        
        mask = torch.bernoulli(torch.full_like(item_index, self.fake_idx_ratio, dtype=torch.float))
        fake_item_idx = torch.where(mask > 0, rand_idx, item_index)
        fake_item_idx[0,:] = 0
        return self.pq_code_embedding(fake_item_idx).mean(dim=-2)

    def seq_item_contrastive_task(self, seq_output, same_pos_id, interaction):
        pos_id = interaction['item_id']
        pos_pq_code = self.pq_codes[pos_id]
        if self.index_assignment_flag:
            pos_items_emb = F.embedding(pos_pq_code, self.reassigned_code_embedding, padding_idx=0).mean(dim=-2)
        else:
            pos_items_emb = self.pq_code_embedding(pos_pq_code).mean(dim=-2)
        pos_items_emb = F.normalize(pos_items_emb, dim=1)

        pos_logits = (seq_output * pos_items_emb).sum(dim=1, keepdim=True) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        fake_item_emb = self.generate_fake_neg_item_emb(pos_pq_code)
        fake_item_emb = F.normalize(fake_item_emb, dim=-1)
        fake_logits = (seq_output * fake_item_emb).sum(dim=1, keepdim=True) / self.temperature
        fake_logits = torch.exp(fake_logits)

        loss = -torch.log(pos_logits / (neg_logits + fake_logits))
        return loss.mean()
    
    def pretrain(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        seq_output = F.normalize(seq_output, dim=1)

        # Remove sequences with the same next item
        pos_id = interaction['item_id']
        same_pos_id = (pos_id.unsqueeze(1) == pos_id.unsqueeze(0))
        same_pos_id = torch.logical_xor(same_pos_id, torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device))

        return self.seq_item_contrastive_task(seq_output, same_pos_id, interaction)
    
    def calculate_loss(self, interaction):
        if self.train_stage == 'pretrain':
            return self.pretrain(interaction)
        
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == 'BPR':
            raise NotImplementedError()
        else:  # self.loss_type = 'CE'
            test_item_emb = self.calculate_item_emb()
            
            if self.temperature > 0:
                seq_output = F.normalize(seq_output, dim=-1)
                test_item_emb = F.normalize(test_item_emb, dim=-1)
            
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            
            if self.temperature > 0:
                logits /= self.temperature
            
            loss = self.loss_fct(logits, pos_items)
            return loss
    def predict(self, interaction):
        raise NotImplementedError()

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.calculate_item_emb()
        
        if self.temperature > 0:
            seq_output = F.normalize(seq_output, dim=-1)
            test_items_emb = F.normalize(test_items_emb, dim=-1)
        
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores