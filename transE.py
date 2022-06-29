import torch
import torch.nn as nn
import torch.nn.functional as F

class MarginLoss(nn.Module):
    def __init__(self, margin=6.0):
        super(MarginLoss, self).__init__()
        self.margin = nn.Parameter(torch.Tensor([margin]))
        self.margin.requires_grad = False

    def forward(self, p_score, n_score):
        return (torch.max(p_score - n_score, -self.margin)).mean() + self.margin

class TransE(nn.Module):
    def __init__(self, args, ent_num = None, rel_num = None):
        super(TransE, self).__init__()
        self.ent_emb = nn.Embedding(ent_num, args.dim)
        self.rel_emb = nn.Embedding(rel_num, args.dim)
        #归一化
        nn.init.uniform_(self.ent_emb.weight.data, -6.0/torch.sqrt(torch.tensor(args.dim)), 6.0/torch.sqrt(torch.tensor(args.dim)))
        nn.init.uniform_(self.rel_emb.weight.data, -6.0 / torch.sqrt(torch.tensor(args.dim)), 6.0 / torch.sqrt(torch.tensor(args.dim)))
        self.norm = args.norm
        self.loss =MarginLoss()
        self.args = args

    def forward(self, batch_corrects=None, batch_corrupts=None, is_eval=False):

        if not is_eval:
            #pos
            c_h = self.ent_emb(batch_corrects[:,0])
            c_t = self.ent_emb(batch_corrects[:,1])
            c_r = self.rel_emb(batch_corrects[:,2])
            #pos_core
            pos_score = c_h + c_r - c_t
            pos_score = torch.norm(pos_score, p=self.args.norm, dim=-1).flatten()
            #neg
            try:
                batch_corrupts[:, 0]
                cu_h = self.ent_emb(batch_corrupts[:, 0])
                cu_t = self.ent_emb(batch_corrupts[:, 1])
                cu_r = self.rel_emb(batch_corrupts[:, 2])
            except Exception as e:
                raise e
            #neg_score
            neg_score = cu_h + cu_r - cu_t
            neg_score = torch.norm(neg_score, p=self.args.norm, dim=-1).flatten()
            loss = self.loss(pos_score, neg_score)
            return loss
        else:
            c_h = self.ent_emb(batch_corrects[:, 0])
            c_t = self.ent_emb(batch_corrects[:, 1])
            c_r = self.rel_emb(batch_corrects[:,2])
            score = torch.norm(c_h+c_r-c_t, p=self.args.norm, dim=-1)
            return score

