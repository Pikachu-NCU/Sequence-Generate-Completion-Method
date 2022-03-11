import torch
from torch import nn
# from emd import emd_module
from utils import square_distance, query_ball_point, index_points
#from PyTorchEMD.emd import earth_mover_distance as emd


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1):
        super(FocalLoss, self).__init__()
        self.gamma, self.alpha = gamma, alpha

    def forward(self, y_pred, y):
        # y_pred: batch_size x num_cls
        # y     : batch_size
        batch_size, num_cls = y_pred.shape[0], y_pred.shape[1]
        y_pred = torch.softmax(y_pred, dim=1).view(-1)[y + torch.arange(0, batch_size).to(y.device) * num_cls]
        loss = -self.alpha * ((1 - y_pred)**self.gamma) * torch.log(y_pred+1e-9)
        return loss.mean(dim=0)


# class EMD(nn.Module):
#     def __init__(self):
#         super(EMD, self).__init__()
#         self.loss_fn = emd
#
#     def forward(self, y_pred, y):
#         return self.loss_fn(y_pred, y, transpose=False)

#
# class EMDList(nn.Module):
#     def __init__(self):
#         super(EMDList, self).__init__()
#         self.loss_fn = EMD()
#
#     def forward(self, pred, gt, test=False):
#         if test:
#             pred_to_gt_mean, gt_to_pred_mean = 0, 0
#             for i in range(len(pred)):
#                 pred_to_gt = self.loss_fn(pred[i].unsqueeze(0), gt[i].unsqueeze(0))
#                 gt_to_pred = self.loss_fn(gt[i].unsqueeze(0), pred[i].unsqueeze(0))
#                 pred_to_gt_mean, gt_to_pred_mean = pred_to_gt_mean+pred_to_gt, gt_to_pred_mean+gt_to_pred
#             pred_to_gt_mean, gt_to_pred_mean = pred_to_gt_mean / len(pred), gt_to_pred_mean / len(pred)
#             return pred_to_gt_mean, gt_to_pred_mean
#         else:
#             loss = 0
#             for i in range(len(pred)):
#                 loss += self.loss_fn(pred[i].unsqueeze(0), gt[i].unsqueeze(0))+self.loss_fn(gt[i].unsqueeze(0), pred[i].unsqueeze(0))
#             return loss / len(pred)
#

class CD(nn.Module):
    def __init__(self, scale=100):
        super(CD, self).__init__()
        # grid scale
        self.scale = scale

    def forward(self, pred, gt, test=False):
        dis = square_distance(pred, gt)
        pred_to_gt = dis.min(dim=2)[0].mean(dim=1)
        gt_to_pred = dis.min(dim=1)[0].mean(dim=1)
        if test:
            return pred_to_gt, gt_to_pred
        else:
            return (0.5*pred_to_gt+0.5*gt_to_pred).mean() * self.scale


class CDList(nn.Module):
    def __init__(self):
        super(CDList, self).__init__()
        self.loss_fn = CD(scale=100)

    def forward(self, pred, gt, test=False):
        # pred: list, have batch point cloud, but their point num is not equal
        # same as above
        if test:
            pred_to_gt_mean, gt_to_pred_mean = 0, 0
            for i in range(len(pred)):
                pred_to_gt, gt_to_pred = self.loss_fn(pred[i].unsqueeze(0), gt[i].unsqueeze(0), True)
                pred_to_gt_mean, gt_to_pred_mean = pred_to_gt_mean+pred_to_gt, gt_to_pred_mean+gt_to_pred
            pred_to_gt_mean, gt_to_pred_mean = pred_to_gt_mean / len(pred), gt_to_pred_mean / len(pred)
            return pred_to_gt_mean*1000, gt_to_pred_mean*1000
        else:
            loss = 0
            for i in range(len(pred)):
                loss += self.loss_fn(pred[i].unsqueeze(0), gt[i].unsqueeze(0))
            return loss / len(pred)


def get_repulsion_loss4(pred, nsample=9, radius=0.07):
    # pred: (batch_size, npoint,3)
    radius = 0.1
    idx = query_ball_point(radius, nsample, pred, pred)

    grouped_pred = index_points(pred, idx)  # (batch_size, npoint, nsample, 3)
    grouped_pred -= pred.unsqueeze(2)

    ##get the uniform loss
    h = 0.12
    dist_square = torch.sum(grouped_pred ** 2, dim=-1)
    dist_square, idx = (-dist_square).topk(5, dim=-1)
    dist_square = -dist_square[:, :, 1:]  # remove the first one
    # dist_square = tf.maximum(1e-12, dist_square)
    dist_square[dist_square < 1e-12] = 1e-12
    dist = torch.sqrt(dist_square)
    weight = torch.exp(-dist_square/h**2)
    uniform_loss = (radius-dist*weight).mean()
    return uniform_loss


class DensityLoss(nn.Module):
    def __init__(self):
        super(DensityLoss, self).__init__()
        self.loss_fn = get_repulsion_loss4

    def forward(self, pred):
        loss = 0
        for i in range(len(pred)):
            loss += self.loss_fn(pred[i].unsqueeze(0))
        return loss / len(pred)


if __name__ == '__main__':
    batch_size, num_cls = 5, 20
    y_pred = torch.rand(5, 20)
    y = torch.rand(batch_size).long()

    loss_fn = FocalLoss(gamma=2, alpha=1)
    loss = loss_fn(y_pred, y)
    print(loss)

