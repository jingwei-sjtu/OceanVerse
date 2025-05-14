import torch
import torch_geometric
import torch.nn as nn

from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
    SparseTensor,
)

from torch_geometric.nn.dense.linear import Linear
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing, GCNConv, GATConv, SAGEConv
from torch_geometric.utils import scatter, mask_feature
import torch.nn.functional as F
import pdb
class MLP(nn.Module):   
    def __init__(self, input_dim, hidden_dim, layer_num=2, activation='relu'):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.activation = activation
        self.build()

    def build(self):
        mlp_layer = [
            nn.Linear(self.input_dim, self.hidden_dim)
        ]
        for _ in range(self.layer_num-1):
            mlp_layer.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.mlp_model = nn.ModuleList(mlp_layer)
    
    def forward(self, input):
        for i in range(self.layer_num):
            output = self.mlp_model[i](input)
            if self.activation == 'relu':
                input = torch.relu(output)
            else:
                input = output
        return output


class STModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_dim, layer_num, time_series_dim, activation='relu'):
        super(STModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.layer_num = layer_num
        self.activation = activation
        self.time_series_dim = time_series_dim
        self.output_dim = output_dim
        self.batch_norm_1 = nn.BatchNorm1d(self.hidden_dim)
        self.batch_norm_2 = nn.BatchNorm1d(self.hidden_dim)
        self.build()
    
    def build(self):
        self.mlp = MLP(self.input_dim , self.hidden_dim, layer_num=2)
        self.temporal_model = nn.LSTM(input_size=self.time_series_dim, hidden_size=16, num_layers = 2, bidirectional = True, batch_first=True)
        gnn_list = [
                OxygenGraphConv(self.hidden_dim*2 + 32, self.hidden_dim, self.edge_dim)
            ]
        if self.layer_num > 1:
            for _ in range(self.layer_num - 1):
                gnn_list.append(OxygenGraphConv(self.hidden_dim, self.hidden_dim, self.edge_dim))
        self.gnn_layer = nn.ModuleList(gnn_list)
        self.output_linear = nn.Linear(self.hidden_dim, self.output_dim)
        self.edge_recons_linear = nn.Linear(2*self.hidden_dim, self.edge_dim)
        self.spatial_decoding = nn.Linear(self.hidden_dim, 2)
        self.geo_encoding = MLP(5, self.hidden_dim, layer_num=2)

    def forward(self, batch):
        x_feature = batch.x.to(torch.float32)
        x_geo = batch.x_geo.to(torch.float32)
        temporal_do = batch.time_series_profile.to(torch.float32)
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr.to(torch.float32)
        x_feature = x_feature.reshape(x_feature.shape[0], -1)
        temporal_do = torch.transpose(temporal_do, 1, 2)
        x_geo_embedding = self.geo_encoding(x_geo)
        profile_feature = self.mlp(x_feature)
        temporal_feature, (_, _) = self.temporal_model(temporal_do)

        gnn_input = torch.cat((profile_feature, x_geo_embedding, temporal_feature[:,5,:]), 1)
        
        for i in range(self.layer_num):
            gnn_output = self.gnn_layer[i](gnn_input, edge_index, edge_attr, x_geo)
            gnn_input = gnn_output
            
        oxygen_pred = self.output_linear(gnn_output)  
        oxygen_pred = torch.clamp(oxygen_pred, min=-0.1, max=1)
        spatial_pred = self.spatial_decoding(gnn_output)
        return oxygen_pred, spatial_pred



class OxygenGraphConv(MessagePassing):
    def __init__(self, input_dim, output_dim, edge_dim):
        super().__init__(aggr='add')
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.batch_norm = nn.BatchNorm1d(self.output_dim)
        self.build()

    def build(self):
        self.feature_transform = Linear(self.input_dim, self.output_dim, bias=False)
        self.edge_transform = Linear(self.edge_dim, 1)
        self.linear = nn.Linear(self.output_dim, self.output_dim)

        self.alpha_transform =  nn.Linear(5, self.input_dim * self.input_dim)
        nn.init.zeros_(self.alpha_transform.weight)
        nn.init.ones_(self.alpha_transform.bias)
        self.beta_transform = nn.Linear(5, self.output_dim)


    def partition_by_physics(self, meta, input):
        alpha = self.alpha_transform(meta).view(-1, self.input_dim, self.input_dim)
        beta = self.beta_transform(meta)
        result = torch.bmm(alpha, input.unsqueeze(2)).squeeze()
        return self.feature_transform(result) + beta 
    
    def forward(self, gnn_input, edge_index, edge_attr, x_geo):

        x = self.partition_by_physics(x_geo, gnn_input)  ## 

        edge_attr = self.edge_transform(edge_attr)
        edge_attr = torch.exp(edge_attr)

        edge_attr_sum = scatter(edge_attr, edge_index[0], reduce='sum')
        edge_attr = edge_attr / edge_attr_sum[edge_index[0]]


        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        out = self.batch_norm(out)
        out = torch.relu(out)
        out = out + x
        return out
    
    def message(self, x_j, edge_attr):
        msg = edge_attr.view(-1, 1) * x_j
        return msg



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        if d_model % 2 == 1:
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, key_padding_mask=src_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
                                     for _ in range(num_layers)])

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


class Baseline_MLP(nn.Module):
    def __init__(self, factor_dim, geo_dim, hidden_dim, time_len, layer_num, edge_dim, meta_dim, profile_len=33):
        super(Baseline_MLP, self).__init__()
        self.profile_len = profile_len
        self.input_dim = geo_dim + factor_dim * profile_len + time_len * profile_len
        self.hidden_dim = hidden_dim
        self.output_dim = profile_len
        self.layer_num = layer_num
        self.token_embedding = nn.Parameter(torch.zeros(self.profile_len, 1))
        self.build()

    def build(self):
        mlp_layer = [
            nn.Linear(self.input_dim, self.hidden_dim)
        ]
        for _ in range(self.layer_num-2):
            mlp_layer.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        mlp_layer.append(nn.Linear(self.hidden_dim, self.output_dim))
        self.mlp_model = nn.ModuleList(mlp_layer)
    
    def forward(self, x_geo, x_factor, do_series, edge_index, edge_attr, x_meta):
        do_series[:, :, int(do_series.shape[-1]/2)] = self.token_embedding.squeeze() # avoid label leakage
        x_factor = x_factor.reshape(x_factor.shape[0], -1)
        do_series = do_series.reshape(do_series.shape[0], -1)
        input_feature = torch.cat((x_geo, x_factor, do_series), dim=-1)
        for i in range(self.layer_num):
            input_feature = torch.relu(self.mlp_model[i](input_feature))
        output = torch.clamp(input_feature, min=-0.1, max=1)
        return output
    
class Baseline_BiLSTM(nn.Module):
    def __init__(self, factor_dim, geo_dim, hidden_dim, time_len, layer_num, edge_dim, meta_dim, profile_len=33):
        super(Baseline_BiLSTM, self).__init__()
        self.time_len = time_len
        self.factor_dim = factor_dim
        self.profile_len = profile_len
        self.geo_dim = geo_dim
        self.geo_encoder = nn.Linear(geo_dim, self.profile_len)
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.token_embedding = nn.Parameter(torch.zeros(self.profile_len, 1))
        d_model = self.factor_dim + self.time_len + 1
        self.temporal_model = nn.LSTM(input_size=d_model, hidden_size=self.hidden_dim, num_layers = self.layer_num, bidirectional = True, batch_first=True)
        self.decoder = nn.Linear(int(hidden_dim * 2), 1)
    def forward(self, x_geo, x_factor, do_series, edge_index, edge_attr, x_meta):
        do_series[:, :, int(do_series.shape[-1]/2)] = self.token_embedding.squeeze() # avoid label leakage
        x_geo = self.geo_encoder(x_geo).unsqueeze(-1)
        series_input = torch.cat((x_factor, do_series, x_geo), dim=-1)
        output, _ = self.temporal_model(series_input)
        output = self.decoder(torch.relu(output)).squeeze(-1)
        return output


class Baseline_Transformer(nn.Module):
    def __init__(self, factor_dim, geo_dim, hidden_dim, time_len, layer_num, edge_dim, meta_dim, profile_len=33):
        super(Baseline_Transformer, self).__init__()
        self.time_len = time_len
        self.factor_dim = factor_dim
        self.profile_len = profile_len
        self.geo_dim = geo_dim
        self.geo_encoder = nn.Linear(geo_dim, self.profile_len)
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.token_embedding = nn.Parameter(torch.zeros(self.profile_len, 1))
        d_model = self.factor_dim + self.time_len + 1
        self.transformer_encoder = TransformerEncoder(self.layer_num, d_model = d_model, num_heads=2, d_ff=self.hidden_dim)
        self.positional_encoding = PositionalEncoding(d_model, max_len=33)
        self.decoder_layer_1 = nn.Linear(d_model, int(d_model * 2))
        self.decoder_layer_2 = nn.Linear(int(d_model * 2), 1)
    def forward(self, x_geo, x_factor, do_series, edge_index, edge_attr, x_meta):
        do_series[:, :, int(do_series.shape[-1]/2)] = self.token_embedding.squeeze() # 避免标签泄漏
        x_geo = self.geo_encoder(x_geo).unsqueeze(-1)
        series_input = torch.cat((x_factor, do_series, x_geo), dim=-1)
        series_input = self.positional_encoding(series_input)
        output = self.transformer_encoder(series_input)
        output = self.decoder_layer_1(torch.relu(output))
        output = self.decoder_layer_2(torch.relu(output)).squeeze(-1)
        return output




    

