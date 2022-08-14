from __future__ import print_function
import torch


class MLPmodel(torch.nn.Module):
    def __init__(self, args, ndatatypes, nreciptypes, nconditions):
        super(MLPmodel, self).__init__()
        self.d_hid = args.nhid
        self.d_emb = args.emsize
        self.nlayers = args.nlayers
        self.n_dtypes = ndatatypes
        self.n_recip = nreciptypes
        self.n_conds = nconditions

        self.datatype_embed = torch.nn.Embedding(self.n_dtypes, self.d_emb)
        self.recipient_embed = torch.nn.Embedding(self.n_recip, self.d_emb)
        self.condition_embed = torch.nn.Embedding(self.n_conds, self.d_emb)
        self.layer1 = torch.nn.Linear(self.d_emb * 3, self.d_hid)
        for n in range(self.nlayers-1):
            setattr(self, "layer{}".format(n+2), torch.nn.Linear(self.d_hid, self.d_hid))
        self.output = torch.nn.Linear(self.d_hid, 2)
        self.dropout = torch.nn.Dropout(args.dropout)

    def init_weights(self):
        initrange = 0.1
        self.datatype_embed.weight.data.uniform_(-initrange, initrange)
        self.recipient_embed.weight.data.uniform_(-initrange, initrange)
        self.condition_embed.weight.data.uniform_(-initrange, initrange)
        for n in range(self.nlayers):
            getattr(self, "layer{}".format(n+1)).weight.data.uniform_(-initrange, initrange)
            getattr(self, "layer{}".format(n+1)).bias.data.zero_()
        self.output.bias.data.zero_()
        self.output.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        datafeas = self.datatype_embed(x[:, 0])
        recip_feats = self.recipient_embed(x[:, 1])
        cond_feats = self.condition_embed(x[:, 2])
        x = torch.cat([datafeas, recip_feats, cond_feats], dim=-1)
        for n in range(self.nlayers):
            x = self.dropout(getattr(self, "layer{}".format(n+1))(x))
            x = torch.relu(x)
        x = self.output(x)
        return x
