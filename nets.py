from utils import *


class MLP(nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(dims[i], dims[i+1]),
                    nn.Tanh(),
                    nn.Dropout(0.3),
                )
                for i in range(len(dims) - 2)
            ],
            nn.Linear(dims[-2], dims[-1])
        )

    def forward(self, x):
        return self.net(x)


# Cross Attention
class CrossAttention(nn.Module):
    def __init__(self, **kwargs):
        super(CrossAttention, self).__init__()
        self.qnet = kwargs.get('qnet', MLP([1024, 256, 1024]))
        self.knet = kwargs.get('knet', MLP([1024, 256, 1024]))
        self.vnet = kwargs.get('vnet', MLP([1024, 256, 1024]))

    # a, b are lists
    # a[i] is tensor of shape N^i_k * 1024 where N^i_k = L^i - k + 1 is the no. of k-mers in seq i from set A
    # b[i] is tensor of shape N^i_k * 1024 where N^i_k = L^i - k + 1 is the no. of k-mers in seq i from set B
    # computes avg_dist (
    #   softmax(q(ai) @ k(bi)^T) v(bi) --> cross attention ai, bi
    #   softmax(q(ai) @ k(bi)^T) v(bi) --> cross attention bi, ai
    # )
    def forward(self, a, b):
        cat_a, cat_b = torch.cat(a, dim=0), torch.cat(b, dim=0)
        qa, ka, va = self.qnet(cat_a), self.knet(cat_a), self.vnet(cat_a)
        qb, kb, vb = self.qnet(cat_b), self.knet(cat_b), self.vnet(cat_b)
        idx_a, idx_b = 0, 0
        emb_a, emb_b = [], []
        for i in range(len(a)):
            idx_a_next, idx_b_next = idx_a + a[i].shape[0], idx_b + b[i].shape[0]
            cross_attn_a = F.softmax(
                qa[idx_a: idx_a_next] / 32. @
                kb[idx_b: idx_b_next].transpose(-2, -1),
                dim=-1
            ) @ vb[idx_b: idx_b_next]
            cross_attn_b = F.softmax(
                qb[idx_b: idx_b_next] / 32. @
                ka[idx_a: idx_a_next].transpose(-2, -1),
                dim=-1
            ) @ va[idx_a: idx_a_next]
            emb_a.append(cross_attn_a.mean(dim=0, keepdim=True))
            emb_b.append(cross_attn_b.mean(dim=0, keepdim=True))
            idx_a, idx_b = idx_a_next, idx_b_next
        return emb_a, emb_b


class CrossAttentionKernel(nn.Module):
    def __init__(self, **kwargs):
        super(CrossAttentionKernel, self).__init__()
        self.cross_attention = CrossAttention(**kwargs)

    def forward(self, a, b):
        emb_a, emb_b = self.cross_attention(a, b)
        return torch.cat([- F.cosine_similarity(emb_a[i], emb_b[i]) for i in range(len(a))], dim=0)


class CrossAttentionPreference(nn.Module):
    def __init__(self, **kwargs):
        super(CrossAttentionPreference, self).__init__()
        self.cross_attention = CrossAttention(**kwargs)

    def forward(self, a, b):
        emb_a, emb_b = self.cross_attention(a, b)
        emb_a, emb_b = torch.cat(emb_a, dim=0), torch.cat(emb_b, dim=0)
        return emb_a - emb_b


# Embedding Layers
class SelfAttentionEmbedding(nn.Module):
    def __init__(self, **kwargs):
        super(SelfAttentionEmbedding, self).__init__()
        self.qnet = kwargs.get('qnet', MLP([1024, 256, 1024]))
        self.knet = kwargs.get('knet', MLP([1024, 256, 1024]))
        self.vnet = kwargs.get('vnet', MLP([1024, 256, 1024]))

    # a is list
    # a[i] is tensor of shape N^i_k * 1024 where N^i_k = L^i - k + 1 is the no. of k-mers in seq i from set A
    def forward(self, a):
        cat_a = torch.cat(a, dim=0)
        qa, ka, va = self.qnet(cat_a), self.knet(cat_a), self.vnet(cat_a)
        idx_a = 0
        res = []
        for i in range(len(a)):
            idx_a_next = idx_a + a[i].shape[0]
            # attn is of shape N^i_k * 1024
            attn = F.softmax(
                qa[idx_a: idx_a_next] / 32. @
                ka[idx_a: idx_a_next].transpose(-1, -2),
                dim=-1
            ) @ va[idx_a: idx_a_next]
            # res[-1] is of shape 1 * 1024
            res.append(attn.mean(dim=0))
        # return batch_size * 1024
        return torch.stack(res, dim=0)


# Multiheaded Linear
class MultiheadLinear(nn.Module):
    def __init__(self, **kwargs):
        super(MultiheadLinear, self).__init__()
        self.qnet = kwargs.get('qnet', MLP([1024, 256, 1024]))
        self.knet = kwargs.get('knet', MLP([1024, 256, 1024]))
        self.vnet = kwargs.get('vnet', MLP([1024, 256, 1024]))

    # a is list
    # a[i] are tensors of shape 1024
    def forward(self, a):
        a = torch.stack(a, dim=0)
        return self.qnet(a) + self.knet(a) + self.vnet(a)
    
class IdentityEmbedding(nn.Module):
    def __init__(self, **kwargs):
        super(IdentityEmbedding, self).__init__()

    def forward(self, a):
        return torch.stack(a, dim=0)


class MultiheadLinearKernel(nn.Module):
    def __init__(self, **kwargs):
        super(MultiheadLinearKernel, self).__init__()
        self.multihead_linear = MultiheadLinear(**kwargs)

    # a, b are lists
    # a[i], b[i] are tensors of shape 1 * 1024
    def forward(self, a, b):
        emb_a, emb_b = self.multihead_linear(a), self.multihead_linear(b)
        return - F.cosine_similarity(emb_a, emb_b)


class MultiheadPreference(nn.Module):
    def __init__(self, **kwargs):
        super(MultiheadPreference, self).__init__()
        self.multihead_linear = MultiheadLinear(**kwargs)

    def forward(self, a, b):
        emb_a, emb_b = self.multihead_linear(a), self.multihead_linear(b)
        return emb_a - emb_b

class IdentityPreference(nn.Module):
    def __init__(self, **kwargs):
        super(IdentityPreference, self).__init__()
    
    def forward(self, a, b):
        a, b = torch.stack(a, dim=0), torch.stack(b, dim=0)
        return a - b

# Simple Euclidean
class EuclideanKernel(nn.Module):
    def __init__(self):
        super(EuclideanKernel, self).__init__()

    # a, b are list
    def forward(self, a, b):
        a, b = torch.cat(a, dim=0), torch.cat(b, dim=0)
        return - F.cosine_similarity(a, b)


