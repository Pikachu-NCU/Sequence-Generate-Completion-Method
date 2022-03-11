import torch
from torch import nn
from utils import *
from torch.nn import functional as F
import math


DownSample = PointConvDensitySetAbstraction


class UpSample(nn.Module):
    def __init__(self, in_channel, mlp, atte=False):
        super(UpSample, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        self.atte = atte
        if self.atte:
            self.atte_layer = MutiHeadAttention(n_head=8, in_channel=128,
                                                qk_channel=64, v_channel=64,
                                                out_channel=128, mid_channel=2048)

    def forward(self, xyz1, xyz2, points1, points2):
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
            if self.atte:
                mask = torch.zeros(B, points1.shape[1], points2.shape[1]).to(torch.device("cuda:0"))
                atte_points = self.atte_layer(points1, points2, points2, mask)
                # print(points1.shape, points2.shape, atte_points.shape)
                new_points = torch.cat([atte_points, new_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points


class MutiHeadAttention(nn.Module):
    def __init__(self, n_head, in_channel, qk_channel, v_channel, out_channel, mid_channel, feedforward=True):
        super(MutiHeadAttention, self).__init__()
        # mutihead attention
        self.n_head = n_head
        self.qk_channel, self.v_channel = qk_channel, v_channel
        self.WQ = nn.Linear(in_channel, qk_channel*n_head, bias=False)
        self.WK = nn.Linear(in_channel, qk_channel*n_head, bias=False)
        self.WV = nn.Linear(in_channel, v_channel*n_head, bias=False)
        self.linear = nn.Linear(v_channel*n_head, out_channel, bias=False)
        # 不确定要不要仿射变换，先不加试试
        self.norm1 = nn.LayerNorm(out_channel, elementwise_affine=False)
        # feedforward
        self.feedforward = feedforward
        if self.feedforward:
            self.feedforward_layer = nn.Sequential(
                nn.Linear(out_channel, mid_channel, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(mid_channel, out_channel, bias=False)
            )
            self.norm2 = nn.LayerNorm(out_channel, elementwise_affine=False)

    def forward(self, query, key, value, mask):
        # q, k, v: batch_size x n x in_channel
        # mask： batch_size x n x n
        batch_size = query.shape[0]
        # batch_size x n x in_channel  -->  batch_size x n x v_channel
        Q = self.WQ(query).view(batch_size, -1, self.n_head, self.qk_channel).transpose(1, 2)  # batch_size x n_head x n x q_channel
        K = self.WK(key).view(batch_size, -1, self.n_head, self.qk_channel).transpose(1, 2)    # batch_size x n_head x n x k_channel
        V = self.WV(value).view(batch_size, -1, self.n_head, self.v_channel).transpose(1, 2)   # batch_size x n_head x n x v_channel
        weight = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.qk_channel)               # batch_size x n_head x n x n
        weight = torch.softmax(weight + mask.unsqueeze(1), dim=3)                              # batch_size x n_head x n x v_channel
        # print(weight.dtype, V.dtype, mask.dtype)
        out = torch.matmul(weight, V).transpose(1, 2).contiguous().view(batch_size, -1, self.n_head*self.v_channel)
        out = self.linear(out)                                                                 # batch_size x n x out_channel
        out = self.norm1(query + out)
        if self.feedforward:
            return self.norm2(out + self.feedforward_layer(out))
        else:
            return out


# ============================== PartSegment ============================
class PartSegTransformer(nn.Module):
    def __init__(self, num_classes, d=128, n_encoder=6, n_head=8, k=30, downsample="PointConv"):
        super(PartSegTransformer, self).__init__()
        # downsample 1
        DownSample = PointConvDensitySetAbstraction if downsample == "PointConv" else PointNetSetAbstraction
        if downsample == "PointNet++ SSG":
            self.ds1 = DownSample(npoint=512, radius=0.2, nsample=32, in_channel=6+3, mlp=[64, 64, 128], group_all=False)
        elif downsample == "PointConv":
            self.ds1 = DownSample(npoint=512, nsample=32, in_channel=6+3, mlp=[64, 64, 128], bandwidth=0.1, group_all=False)

        self.encoders_1 = nn.ModuleList([
            MutiHeadAttention(
                n_head,
                in_channel=d,
                qk_channel=64,
                v_channel=64,
                out_channel=d,
                mid_channel=2048
            ) for _ in range(n_encoder // 2)
        ])

        # downsample 2
        if downsample == "PointNet++ SSG":
            self.ds2 = DownSample(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        elif downsample == "PointConv":
            self.ds2 = DownSample(npoint=128, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], bandwidth=0.2, group_all=False)

        self.encoders_2 = nn.ModuleList([
            MutiHeadAttention(
                n_head,
                in_channel=d*2,
                qk_channel=64,
                v_channel=64,
                out_channel=d*2,
                mid_channel=2048
            ) for _ in range(n_encoder // 2)
        ])

        # downsample 3
        if downsample == "PointNet++ SSG":
            self.ds3 = DownSample(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        elif downsample == "PointConv":
            self.ds3 = DownSample(npoint=1, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], bandwidth=0.4, group_all=True)

        self.up3 = UpSample(in_channel=1280, mlp=[256, 128])
        self.up2 = UpSample(in_channel=384, mlp=[256, 128], atte=True)
        self.up1 = UpSample(in_channel=128 + 16 + 6 + 3, mlp=[128, 128])

        self.fc = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(128, num_classes, 1)
        )
        self.k = k
        self.num_classes = num_classes

    def forward(self, x, cls_):
        inf = -1e9
        x = x.permute([0, 2, 1])
        l0_points, l0_xyz = x, x[:, :3, :]
        l1_xyz, l1_points = self.ds1(l0_xyz, l0_points)                         # batch_size x n x 3, batch_size x n x d
        l1_points = l1_points.permute([0, 2, 1])
        batch_size, n = l1_points.shape[0], l1_points.shape[1]

        # 把特征空间距离太大的mask掉
        def make_mask(features, n):
            dis = square_distance(features, features)  # batch_size x n x n
            top_k_idx = dis.topk(self.k, dim=2, largest=False, sorted=False)[1]  # batch_size x n x k
            mask = torch.ones(batch_size, n, n).view(-1).to(x.device)
            mask[((torch.arange(0, batch_size * n) * n).unsqueeze(1).to(x.device) + top_k_idx.view(-1, self.k)).view(-1)] = 0
            mask = mask.view(batch_size, n, n) * inf
            return mask
        # self-attention
        for encoder in self.encoders_1:
            mask = make_mask(l1_points, n)
            l1_points = encoder(l1_points, l1_points, l1_points, mask)
        # downsample
        l1_points = l1_points.permute([0, 2, 1])
        l2_xyz, l2_points = self.ds2(l1_xyz, l1_points)
        l2_points = l2_points.permute([0, 2, 1])
        batch_size, n = l2_points.shape[0], l2_points.shape[1]
        # self-attention
        for encoder in self.encoders_2:
            mask = make_mask(l2_points, n)
            l2_points = encoder(l2_points, l2_points, l2_points, mask)
        # global feature
        l2_points = l2_points.permute([0, 2, 1])
        l3_xyz, l3_points = self.ds3(l2_xyz, l2_points)
        # upsample
        l2_points = self.up3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.up2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = cls_.view(batch_size, 16, 1).repeat(1, 1, l0_xyz.shape[2])
        l0_points = self.up1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot, l0_xyz, l0_points], 1), l1_points)
        #print(self.fc(l0_points).permute([0, 2, 1]).contiguous().view(-1, self.num_classes).shape)
        return self.fc(l0_points).permute([0, 2, 1]).contiguous().view(-1, self.num_classes)


# ============================== Classify ============================
class ClsTransformer(nn.Module):
    def __init__(self, num_classes, d=128, n_encoder=6, n_head=8, k=30, downsample="PointConv"):
        super(ClsTransformer, self).__init__()
        # downsample 1
        DownSample = PointConvDensitySetAbstraction if downsample == "PointConv" else PointNetSetAbstraction
        if downsample == "PointNet++ SSG":
            self.ds1 = DownSample(npoint=512, radius=0.2, nsample=32, in_channel=3+6, mlp=[64, 64, 128], group_all=False)
        elif downsample == "PointConv":
            self.ds1 = DownSample(npoint=512, nsample=32, in_channel=3+6, mlp=[64, 128], bandwidth=0.1, group_all=False)

        self.encoders_1 = nn.ModuleList([
            MutiHeadAttention(
                n_head,
                in_channel=d,
                qk_channel=64,
                v_channel=64,
                out_channel=d,
                mid_channel=2048
            ) for _ in range(n_encoder // 2)
        ])

        # downsample 2
        if downsample == "PointNet++ SSG":
            self.ds2 = DownSample(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        elif downsample == "PointConv":
            self.ds2 = DownSample(npoint=128, nsample=64, in_channel=128 + 3, mlp=[128, 256], bandwidth=0.2, group_all=False)

        self.encoders_2 = nn.ModuleList([
            MutiHeadAttention(
                n_head,
                in_channel=d*2,
                qk_channel=64,
                v_channel=64,
                out_channel=d*2,
                mid_channel=2048
            ) for _ in range(n_encoder // 2)
        ])

        # downsample 3
        if downsample == "PointNet++ SSG":
            self.ds3 = DownSample(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        elif downsample == "PointConv":
            self.ds3 = DownSample(npoint=1, nsample=None, in_channel=256 + 3, mlp=[256, 1024], bandwidth=0.4, group_all=True)

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
        self.k = k
        self.num_classes = num_classes

    def forward(self, x):
        inf = -1e9
        x = x.permute([0, 2, 1])
        l0_points, l0_xyz = x, x[:, :3, :]
        l1_xyz, l1_points = self.ds1(l0_xyz, l0_points)                         # batch_size x n x 3, batch_size x n x d
        l1_points = l1_points.permute([0, 2, 1])
        batch_size, n = l1_points.shape[0], l1_points.shape[1]

        # 把特征空间距离太大的mask掉
        def make_mask(features, n):
            dis = square_distance(features, features)  # batch_size x n x n
            top_k_idx = dis.topk(self.k, dim=2, largest=False, sorted=False)[1]  # batch_size x n x k
            mask = torch.ones(batch_size, n, n).view(-1).to(x.device)
            mask[((torch.arange(0, batch_size * n) * n).unsqueeze(1).to(x.device) + top_k_idx.view(-1, self.k)).view(-1)] = 0
            mask = mask.view(batch_size, n, n) * inf
            return mask
        # self-attention
        for encoder in self.encoders_1:
            mask = make_mask(l1_points, n)
            l1_points = encoder(l1_points, l1_points, l1_points, mask)
        # downsample
        l1_points = l1_points.permute([0, 2, 1])
        l2_xyz, l2_points = self.ds2(l1_xyz, l1_points)
        l2_points = l2_points.permute([0, 2, 1])
        batch_size, n = l2_points.shape[0], l2_points.shape[1]
        # self-attention
        for encoder in self.encoders_2:
            mask = make_mask(l2_points, n)
            l2_points = encoder(l2_points, l2_points, l2_points, mask)
        # global feature
        l2_points = l2_points.permute([0, 2, 1])
        l3_xyz, l3_points = self.ds3(l2_xyz, l2_points)
        #print(self.fc(l3_points.view(x.shape[0], -1)).shape)
        return self.fc(l3_points.view(x.shape[0], -1))


# ============================  Completion =================================
class CompletionTransformer(nn.Module):
    def __init__(self, n_shift_points=4, d=128, n_encoder=6, n_head=8, k=30, downsample="PointConv"):
        super(CompletionTransformer, self).__init__()
        # downsample 1
        DownSample = PointConvDensitySetAbstraction if downsample == "PointConv" else PointNetSetAbstraction
        if downsample == "PointNet++ SSG":
            self.ds1 = DownSample(npoint=512, radius=0.2, nsample=32, in_channel=3, mlp=[64, 64, 128], group_all=False)
        elif downsample == "PointConv":
            self.ds1 = DownSample(npoint=512, nsample=32, in_channel=3, mlp=[64, 64, 128], bandwidth=0.1, group_all=False)

        self.encoders_1 = nn.ModuleList([
            MutiHeadAttention(
                n_head,
                in_channel=d,
                qk_channel=64,
                v_channel=64,
                out_channel=d,
                mid_channel=2048
            ) for _ in range(n_encoder // 2)
        ])

        # downsample 2
        if downsample == "PointNet++ SSG":
            self.ds2 = DownSample(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], group_all=False)
        elif downsample == "PointConv":
            self.ds2 = DownSample(npoint=128, nsample=64, in_channel=128 + 3, mlp=[128, 128, 256], bandwidth=0.2, group_all=False)

        self.encoders_2 = nn.ModuleList([
            MutiHeadAttention(
                n_head,
                in_channel=d*2,
                qk_channel=64,
                v_channel=64,
                out_channel=d*2,
                mid_channel=2048
            ) for _ in range(n_encoder // 2)
        ])

        # downsample 3
        if downsample == "PointNet++ SSG":
            self.ds3 = DownSample(npoint=None, radius=None, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], group_all=True)
        elif downsample == "PointConv":
            self.ds3 = DownSample(npoint=1, nsample=None, in_channel=256 + 3, mlp=[256, 512, 1024], bandwidth=0.4, group_all=True)

        self.up3 = UpSample(in_channel=1280, mlp=[256, 128])
        self.up2 = UpSample(in_channel=384, mlp=[256, 128], atte=True)
        self.up1 = UpSample(in_channel=128 + 16 + 0 + 3, mlp=[128, 128])

        self.fc = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Conv1d(128, n_shift_points*3, 1),


        )
        self.k = k
        self.n_shift_points = n_shift_points
        ch = 405 * self.n_shift_points * 3
        self.linear = torch.nn.Linear(ch, ch)

    def forward(self, x, cls_, need_shift_num):
        inf = -1e9
        x = x.permute([0, 2, 1])
        l0_points, l0_xyz = None, x
        l1_xyz, l1_points = self.ds1(l0_xyz, l0_points)                         # batch_size x n x 3, batch_size x n x d
        l1_points = l1_points.permute([0, 2, 1])
        batch_size, n = l1_points.shape[0], l1_points.shape[1]

        # 把特征空间距离太大的mask掉
        def make_mask(features, n):
            dis = square_distance(features, features)  # batch_size x n x n
            top_k_idx = dis.topk(self.k, dim=2, largest=False, sorted=False)[1]  # batch_size x n x k
            mask = torch.ones(batch_size, n, n).view(-1).to(x.device)
            mask[((torch.arange(0, batch_size * n) * n).unsqueeze(1).to(x.device) + top_k_idx.view(-1, self.k)).view(-1)] = 0
            mask = mask.view(batch_size, n, n) * inf
            return mask
        # self-attention
        for encoder in self.encoders_1:
            mask = make_mask(l1_points, n)
            l1_points = encoder(l1_points, l1_points, l1_points, mask)
        # downsample
        l1_points = l1_points.permute([0, 2, 1])
        l2_xyz, l2_points = self.ds2(l1_xyz, l1_points)
        l2_points = l2_points.permute([0, 2, 1])
        batch_size, n = l2_points.shape[0], l2_points.shape[1]
        # self-attention
        for encoder in self.encoders_2:
            mask = make_mask(l2_points, n)
            l2_points = encoder(l2_points, l2_points, l2_points, mask)
        # global feature
        l2_points = l2_points.permute([0, 2, 1])
        l3_xyz, l3_points = self.ds3(l2_xyz, l2_points)
        # print(l0_points, l1_points.shape, l2_points.shape, l2_points.shape, )
        # upsample
        l2_points = self.up3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.up2(l1_xyz, l2_xyz, l1_points, l2_points)
        cls_label_one_hot = cls_.view(batch_size, 16, 1).repeat(1, 1, l0_xyz.shape[2])
        l0_points = self.up1(l0_xyz, l1_xyz, torch.cat([cls_label_one_hot, l0_xyz], 1), l1_points)
        shift_dis = self.fc(l0_points).permute([0, 2, 1])  # batch_size x n x shift_points_num

        shift_dis = shift_dis[:, -405:, :]
        shift_dis = shift_dis.reshape(shift_dis.shape[0], -1)
        shift_dis = self.linear(shift_dis)
        shift_dis = shift_dis.reshape(shift_dis.shape[0], -1, self.n_shift_points*3)

        shift_result = []
        x = x.permute([0, 2, 1])
        for i in range(shift_dis.shape[0]):

            cur_shift_dis = shift_dis[i, -need_shift_num[i]:, :].contiguous().view(need_shift_num[i], -1, 3)
            #print(cur_shift_dis.shape)
            # need_shift_num x n_shift_points x 3
            shifted_points = x[i, x.shape[1]-need_shift_num[i]:, :].contiguous().view(-1, 1, 3).repeat([1, 7, 1])+cur_shift_dis
            shift_result.append(shifted_points.view(-1, 3))
        return shift_result


# =================== Transformer ======================
def get_mask(seq1, seq2, pad_num):
    # seq1: [
    #     [1, 5, 3, 4, 0],  句子1
    #     [2, 4, 9, 0, 0],  句子2
    #     [1, 4, 7, 8, 8]   句子3
    # ] 每句n个词
    # seq2: [
    #     [1, 4, 0],
    #     [2, 0, 0],
    #     [3, 6, 0]
    # ] 每句m个词
    # 返回 batch_size x n x m的mask矩阵，有用的地方是0，没用的地方是负无穷，负无穷在softmax后会变为0
    batch_size, n = seq1.shape
    _, m = seq2.shape
    inf = 1e9
    mask = seq1.eq(pad_num).unsqueeze(2).expand(batch_size, n, m) | seq2.eq(pad_num).unsqueeze(1).expand(batch_size, n, m)
    return mask*(-inf)


def get_decoder_mask(seq, pad_num):
    # seq: batch_size x n
    inf = 1e9
    # 解码器不能预知未来，所以要再加个上三角mask
    mask = get_mask(seq, seq, pad_num)
    mask = mask + torch.FloatTensor(np.triu(np.ones(shape=(seq.shape[1], seq.shape[1])), 1)*(-inf)).to(mask.device)
    mask[mask != 0] = -inf
    return mask


class PositionEncode(nn.Module):
    def __init__(self, d, dropout=0.1, max_len=5000):
        super(PositionEncode, self).__init__()
        self.dp = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dp(x)


class Encoder(nn.Module):
    def __init__(self, voc_size, d=512, n_encoder=6, n_head=8, pad_num=None):
        super(Encoder, self).__init__()
        self.inp_emd = nn.Embedding(voc_size, d)
        self.pos_emd = PositionEncode(d)
        self.encoders = nn.ModuleList([
            MutiHeadAttention(
                n_head,
                in_channel=d,
                qk_channel=64,
                v_channel=64,
                out_channel=d,
                mid_channel=2048
            ) for _ in range(n_encoder)
        ])
        self.pad_num = pad_num

    def forward(self, x):
        # x: batch_size x n
        mask = get_mask(x, x, self.pad_num)
        x = self.inp_emd(x)                                  # batch_size x n x d
        x = self.pos_emd(x.transpose(0, 1)).transpose(0, 1)  # batch_size x n x d
        for encoder in self.encoders:
            x = encoder(x, x, x, mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, n_head, in_channel, qk_channel, v_channel, out_channel, mid_channel):
        super(DecoderLayer, self).__init__()
        self.att_self = MutiHeadAttention(n_head, in_channel, qk_channel, v_channel, out_channel, mid_channel, False)
        self.att_encode = MutiHeadAttention(n_head, in_channel, qk_channel, v_channel, out_channel, mid_channel)

    def forward(self, encoder_output, decoder_input, mask_encode, mask_self):
        # mask_encoder是和encoder的输出做self attention，只需把自己和编码器输出的pad部分遮掉就行，不用上三角mask
        # mask_self是自己与自己做self attention，防止预知未来要加上三角mask
        decoder_out = self.att_self(decoder_input, decoder_input, decoder_input, mask_self)
        decoder_out = self.att_encode(decoder_out, encoder_output, encoder_output, mask_encode)
        return decoder_out


class Decoder(nn.Module):
    def __init__(self, voc_size, out_dim, d=512, n_decoder=6, n_head=8, pad_num=None):
        super(Decoder, self).__init__()
        self.inp_emd = nn.Embedding(voc_size, d)
        self.pos_emd = PositionEncode(d)
        self.decoders = nn.ModuleList([
            DecoderLayer(
                n_head,
                in_channel=d,
                qk_channel=64,
                v_channel=64,
                out_channel=d,
                mid_channel=2048
            ) for _ in range(n_decoder)
        ])
        self.fc = nn.Linear(d, out_dim)
        self.pad_num = pad_num

    def forward(self, decoder_inp, encoder_out, mask_encode):
        # decoder_inp: batch_size x m
        # encoder_out: batch_size x n x d
        # mask_encode: batch_size x m x n
        mask_self = get_decoder_mask(decoder_inp, self.pad_num)                  # batch_size x m x m
        decoder_inp = self.inp_emd(decoder_inp)
        decoder_inp = self.pos_emd(decoder_inp.transpose(0, 1)).transpose(0, 1)  # batch_size x m x d
        for i, decoder in enumerate(self.decoders):
            decoder_inp = decoder(encoder_out, decoder_inp, mask_encode, mask_self)
        return self.fc(decoder_inp)


class Transformer(nn.Module):
    def __init__(self, inp_voc_size, out_voc_size, out_dim, d=512, n_encoder=6, n_encoder_head=8, n_decoder=6, n_decoder_head=8, pad_num=None):
        super(Transformer, self).__init__()
        self.encoder = Encoder(inp_voc_size, d, n_encoder_head, n_encoder, pad_num)
        self.decoder = Decoder(out_voc_size, out_dim, d, n_decoder_head, n_decoder, pad_num)
        self.pad_num = pad_num

    def forward(self, inp, out):
        # inp是输入序列，out是inp对应的输出序列
        mask_encoder = get_mask(out, inp, pad_num=self.pad_num)
        encoder_output = self.encoder(inp)
        decoder_output = self.decoder(out, encoder_output, mask_encoder)
        return decoder_output


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.rand(2, 1024, 6).to(device)
    model = PartSegTransformer(30, downsample="PointConv")
    model.to(device)
    y = model(x, torch.rand(2, 16).to(device))
    print(y.shape)