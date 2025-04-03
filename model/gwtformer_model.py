# model/gwtformer_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class nconv(nn.Module):
    """Graph convolution based on matrix multiplication (for adjacency)."""
    def __init__(self):
        super(nconv, self).__init__()
    def forward(self, x, A):
        # x: (batch, c, v, l), A: (v, v) adjacency matrix
        # Perform nconv: summation over neighbors -> einsum does (batch, c, v, l) * (v, w) -> (batch, c, w, l)
        x = torch.einsum('ncvl,vw->ncwl', x, A)
        return x.contiguous()

class linear(nn.Module):
    """Simple linear (1x1 convolution) layer to change feature dimensions."""
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        # 1x1 convolution to map from c_in channels to c_out channels
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1,1))
    def forward(self, x):
        return self.mlp(x)

class gcn(nn.Module):
    """
    Graph Convolution Network (GCN) with support for adaptive adjacency.
    - support_len: number of support adjacency matrices (e.g., 2 for double transition).
    - order: propagation steps to include (e.g., 2 for up to 2-hop neighbors).
    """
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv()
        # Effective input channels = (order * support_len + 1) * c_in 
        # (+1 for the original features without convolution)
        c_in_eff = (order * support_len + 1) * c_in
        self.mlp = linear(c_in_eff, c_out)
        self.dropout = dropout
        self.order = order
    def forward(self, x, supports):
        # supports: list of adjacency matrices to use
        out = [x]  # include original x (order 0)
        # Aggregate neighbors for each support matrix
        for A in supports:
            x1 = self.nconv(x, A)           # 1-hop neighbors
            out.append(x1)
            for k in range(2, self.order+1):
                # Higher-order neighbors (k-hop)
                x1 = self.nconv(x1, A)
                out.append(x1)
        # Concatenate along the channel dimension
        h = torch.cat(out, dim=1)
        # Apply linear layer and dropout
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h

class TemporalTransformerLayer(nn.Module):
    """
    Temporal Transformer layer: applies multi-head self-attention across the time dimension
    for each node's time series, followed by a position-wise feed-forward network.
    """
    def __init__(self, c_in, num_heads, ffn_hidden, dropout):
        super(TemporalTransformerLayer, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=c_in, num_heads=num_heads, dropout=dropout, batch_first=True)
        # Position-wise Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(c_in, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, c_in)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(c_in)
        self.norm2 = nn.LayerNorm(c_in)
    def forward(self, x):
        # x: (batch, c_in, num_nodes, seq_len)
        N, C, V, L = x.shape
        # Prepare input for multi-head attention: treat each node's time series as a separate sequence in the batch
        x_seq = x.permute(0, 2, 3, 1).reshape(N * V, L, C)  # (batch*num_nodes, seq_len, c_in)
        # Self-attention (Q = K = V = x_seq). The output has the same shape as input.
        attn_out, _ = self.attn(x_seq, x_seq, x_seq)       # (N*V, seq_len, C)
        # Add & Norm
        x_seq = self.norm1(x_seq + self.dropout(attn_out))
        # Feed-Forward Network
        ffn_out = self.ffn(x_seq)                         # (N*V, seq_len, C)
        # Add & Norm
        x_seq = self.norm2(x_seq + self.dropout(ffn_out)) 
        # Reshape back to (batch, C, V, L)
        x_out = x_seq.reshape(N, V, L, C).permute(0, 3, 1, 2).contiguous()
        return x_out

class GWTformer(nn.Module):
    """
    GWTformer model: Combines graph convolutions with temporal Transformer layers.
    Replaces the TCN (temporal conv) in Graph WaveNet with self-attention while preserving spatial graph conv.
    """
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
        # By default, use ffn_dim = 2 * residual_channels if not provided
        if ffn_dim is None:
            ffn_dim = residual_channels * 2
        # Initialize support adjacencies
        self.supports = [] if supports is None else supports
        self.supports_len = len(self.supports)
        # Adaptive adjacency: learnable node embedding for additional graph structure
        if self.gcn_bool and self.addaptadj:
            if aptinit is None:
                # Randomly initialize adaptive adjacency (node embeddings)
                if supports is None:
                    self.supports = []
                self.nodevec1 = nn.Parameter(torch.randn(num_nodes, 10).to(device))  # first node embedding
                self.nodevec2 = nn.Parameter(torch.randn(10, num_nodes).to(device))  # second node embedding
                self.supports_len += 1  # adaptive adj will be added as an extra support
            else:
                # Initialize adaptive adj using SVD of given adjacency (aptinit)
                if supports is None:
                    self.supports = []
                m, p, n = torch.svd(aptinit)
                # Use top-10 singular vectors for initialization
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10]**0.5))
                initemb2 = torch.mm(torch.diag(p[:10]**0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(initemb1, requires_grad=True).to(device)
                self.nodevec2 = nn.Parameter(initemb2, requires_grad=True).to(device)
                self.supports_len += 1
        # Input transformation: 1x1 convolution to project input features to residual_channels
        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1,1))
        # Define module lists for temporal and spatial layers
        self.tfn_layers = nn.ModuleList()       # list of TemporalTransformerLayer modules
        self.skip_convs = nn.ModuleList()       # list of skip connection 1x1 convs
        self.gconv = nn.ModuleList()            # list of graph conv layers (if gcn_bool is True)
        self.residual_convs = nn.ModuleList()   # list of 1x1 conv for residual (if gcn_bool is False)
        self.bn = nn.ModuleList()               # list of batch norms for each layer
        # Compute receptive field (same logic as Graph WaveNet) for padding
        self.receptive_field = 1
        for b in range(blocks):
            additional_scope = kernel_size - 1  # = 1 for kernel_size=2
            for i in range(layers):
                # Temporal Transformer layer replaces the dilated conv layer
                self.tfn_layers.append(TemporalTransformerLayer(residual_channels, num_heads, ffn_dim, dropout))
                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=skip_channels, kernel_size=(1,1)))
                if self.gcn_bool:
                    # Graph convolution layer for spatial dependency
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.supports_len))
                else:
                    # If no graph conv, use a 1x1 conv to maintain residual dimensions
                    self.residual_convs.append(nn.Conv2d(in_channels=dilation_channels, out_channels=residual_channels, kernel_size=(1,1)))
                # Batch norm layer
                self.bn.append(nn.BatchNorm2d(residual_channels))
                # Update receptive field size
                self.receptive_field += additional_scope
                additional_scope *= 2  # dilation doubles each layer in original WaveNet logic
        # Output projection layers
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels, out_channels=end_channels, kernel_size=(1,1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1,1), bias=True)
    
    def forward(self, input_data):
        """
        Forward pass of GWTformer.
        input_data: (batch, in_dim, num_nodes, seq_len)
        Returns: (batch, out_dim, num_nodes, 1)
        """
        # Pad the input sequence length to match the receptive field, so that we can make predictions for all horizon steps
        seq_len = input_data.size(3)
        if seq_len < self.receptive_field:
            input_data = F.pad(input_data, (self.receptive_field - seq_len, 0, 0, 0))
        # Initial projection
        x = self.start_conv(input_data)  # shape: (batch, residual_channels, num_nodes, seq_len_padded)
        skip = 0  # skip connection accumulator
        # Compute adaptive adjacency matrix (only once per forward) if enabled
        if self.gcn_bool and self.addaptadj and len(self.supports) > 0:
            # Generate adaptive adj matrix via node embeddings
            adp = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)  # (num_nodes, num_nodes)
            supports = self.supports + [adp]  # combine static and adaptive supports
        else:
            supports = self.supports
        # Temporal + Spatial blocks
        for i in range(len(self.tfn_layers)):
            residual = x  # save residual
            # Temporal self-attention + FFN
            x = self.tfn_layers[i](x)        # (batch, residual_channels, num_nodes, seq_len_padded)
            # Skip connection: take the last time step output of this layer
            s = x[..., -1:]                 # shape: (batch, residual_channels, num_nodes, 1)
            s = self.skip_convs[i](s)       # map to skip_channels
            skip = s if (isinstance(skip, int)) else (skip + s)  # accumulate skip outputs
            # Spatial graph convolution (or simple residual conv)
            if self.gcn_bool and supports:
                x = self.gconv[i](x, supports)
            else:
                x = self.residual_convs[i](x)
            # Residual connection (align time dimension)
            x = x + residual[..., -x.size(3):]
            # Batch normalization
            x = self.bn[i](x)
        # After all layers, apply output conv layers on the accumulated skip connection
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))      # shape: (batch, end_channels, num_nodes, 1)
        x = self.end_conv_2(x)             # shape: (batch, out_dim, num_nodes, 1)
        return x

