# model
import torch
import torch.nn as nn
import torch.nn.functional as F

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()
    def forward(self, x, A):
        x = torch.einsum('ncvl,vw->ncwl', x, A)
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1,1))
    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        c_in_eff = (order * support_len + 1) * c_in
        self.mlp = linear(c_in_eff, c_out)
        self.dropout = dropout
        self.order = order
    def forward(self, x, supports):
        out = [x]
        for A in supports:
            x1 = self.nconv(x, A)
            out.append(x1)
            for k in range(2, self.order+1):
                x1 = self.nconv(x1, A)
                out.append(x1)
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class TemporalTransformerLayer(nn.Module):
    def __init__(self, c_in, num_heads, ffn_hidden, dropout):
        super(TemporalTransformerLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=c_in, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(c_in, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, c_in))
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(c_in)
        self.norm2 = nn.LayerNorm(c_in)
    def forward(self, x):
        N, C, V, L = x.shape
        x_seq = x.permute(0, 2, 3, 1).reshape(N * V, L, C)
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)
        x_seq = self.norm1(x_seq + self.dropout(attn_out))
        ffn_out = self.ffn(x_seq)
        x_seq = self.norm2(x_seq + self.dropout(ffn_out))
        x_out = x_seq.reshape(N, V, L, C).permute(0, 3, 1, 2).contiguous()
        return x_out

class GWTformer(nn.Module):
    def __init__(self, device, num_nodes, dropout=0.3,
                 supports=None, gcn_bool=True, addaptadj=True, aptinit=None,
                 in_dim=2, out_dim=12, residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=512, kernel_size=2, blocks=4, layers=2,
                 num_heads=4, ffn_dim=None):
        super(GWTformer, self).__init__()
        self.dropout = dropout
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.num_nodes = num_nodes
        if ffn_dim is None:
            ffn_dim = residual_channels * 2
        self.supports = [] if supports is None else supports
        self.supports_len = len(self.supports)
        if self.gcn_bool and self.addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device))
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device))
                self.supports_len += 1
            else:
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10]**0.5))
                initemb2 = torch.mm(torch.diag(p[:10]**0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1,1))
        self.tfn_layers = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.receptive_field = 1
        for b in range(blocks):
            additional_scope = kernel_size - 1
            for i in range(layers):
                self.tfn_layers.append(TemporalTransformerLayer(residual_channels, num_heads, ffn_dim, dropout))
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=skip_channels, kernel_size=(1,1)))
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))
                else:
                    self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=residual_channels, kernel_size=(1,1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                self.receptive_field += additional_scope
                additional_scope *= 2
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1,1), bias=True)

    def forward(self, input_data):
        seq_len = input_data.size(3)
        if seq_len < self.receptive_field:
            input_data = F.pad(input_data, (self.receptive_field - seq_len, 0, 0, 0))
        x = self.start_conv(input_data)
        skip = 0
        if self.gcn_bool and self.addaptadj and len(self.supports) > 0:
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            supports = self.supports + [adp]
        else:
            supports = self.supports
        for i in range(len(self.tfn_layers)):
            residual = x
            x = self.tfn_layers[i](x)
            s = x[..., -1:]
            s = self.skip_convs[i](s)
            skip = s if (isinstance(skip, int)) else (skip + s)
            if self.gcn_bool and supports:
                x = self.gconv[i](x, supports)
            else:
                x = self.residual_convs[i](x)
            x = x + residual[..., -x.size(3):]
            x = self.bn[i](x)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
