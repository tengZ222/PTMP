import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.nn.functional import relu
import copy
#（seq_len,batch,f_in）
class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_head, f_in, f_out, attn_dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.n_head = n_head
        self.f_in = f_in
        self.f_out = f_out
        self.w = nn.Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = nn.Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = nn.Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(attn_dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    # （seq_len,batch,f_in）
    def forward(self, h):
        bs, n = h.size()[:2]
        h_prime = torch.matmul(h.unsqueeze(1), self.w)
        attn_src = torch.matmul(h_prime, self.a_src)
        attn_dst = torch.matmul(h_prime, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        if self.bias is not None:
            return output + self.bias, attn
        else:
            return output, attn

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.n_head)
            + " -> "
            + str(self.f_in)
            + " -> "
            + str(self.f_out)
            + ")"
        )
#n_units = [128, 64, 32, 128]
#n_heads = [4,4,1]

class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout=0.2):
        super(GAT, self).__init__()
        self.n_layer = len(n_units) - 1
        self.dropout = dropout
        self.layer_stack = nn.ModuleList()

        for i in range(self.n_layer):
            f_in = n_units[i] * n_heads[i - 1] if i or i == 2 else n_units[i]
            self.layer_stack.append(
                BatchMultiHeadGraphAttention(
                    n_heads[i], f_in=f_in, f_out=n_units[i + 1], attn_dropout=dropout
                )
            )
         #[4,128,64]
         #[4,256,32]
         #[1,128,128]
        self.norm_list = [
            torch.nn.InstanceNorm1d(128).cuda(),
            torch.nn.InstanceNorm1d(512).cuda(),
            torch.nn.InstanceNorm1d(512).cuda(),
        ]

    def forward(self, x):
        bs, n = x.size()[:2]  #（seq_len,batch,f_in）
        for i, gat_layer in enumerate(self.layer_stack):
            x = self.norm_list[i](x.permute(0, 2, 1)).permute(0, 2, 1)#（seq_len,batch,f_in）
            x, attn = gat_layer(x)  #output, attn
            if i + 1 == self.n_layer:
                x = x.squeeze(dim=1)
            else:
                x = F.elu(x.transpose(1, 2).contiguous().view(bs, n, -1))
                x = F.dropout(x, self.dropout, training=self.training)
        else:
            return x


class GAT_Encoder(nn.Module):
    def __init__(self):
        super(GAT_Encoder, self).__init__()
        n_units = [128, 64, 32, 128]
        n_heads = [4, 4, 1]
        dropout = 0.2
        self.n_units = n_units
        self.gat_net = GAT(n_units, n_heads, dropout)

    def forward(self, combined):  #[B, K*N, embed_size]


        curr_seq_embedding_traj = combined.transpose(0,1)
        curr_seq_graph_embedding = self.gat_net(curr_seq_embedding_traj)
        curr_seq_graph_embedding = curr_seq_graph_embedding.transpose(0, 1)

        return curr_seq_graph_embedding



























        """
        obs_traj_embedding= obs_traj_embedding_and_seq_start_end[0]
        seq_start_end = obs_traj_embedding_and_seq_start_end[1]

        for start, end in seq_start_end.data:

            #输入obs_traj_embedding的形状为（batch,seq_len,f_in）
            #将seq_len和batch的顺序调换
            #输出后形状为（seq_len,batch,f_out）

            curr_seq_embedding_traj = obs_traj_embedding[start:end, :, :].transpose(0,1)
            curr_seq_graph_embedding = self.gat_net(curr_seq_embedding_traj)


            #将形状变换为(batch,seq_len,f_out)

            curr_seq_graph_embedding = curr_seq_graph_embedding.transpose(0, 1)
            graph_embeded_data.append(curr_seq_graph_embedding)
        """