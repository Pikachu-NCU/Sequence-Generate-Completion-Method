from utils import *
import torch


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


if __name__ == '__main__':
    cd_fn = CD()
    fps1 = np.loadtxt("C:/Users/sdnyz/Desktop/新建文件夹 (2)/Grid_Fps_1.txt", delimiter=';')
    fps2 = np.loadtxt("C:/Users/sdnyz/Desktop/新建文件夹 (2)/Grid_Fps_19.txt", delimiter=';')
    g1 = np.loadtxt("C:/Users/sdnyz/Desktop/新建文件夹 (2)/Grid_1.txt", delimiter=';')
    g2 = np.loadtxt("C:/Users/sdnyz/Desktop/新建文件夹 (2)/Grid_19.txt", delimiter=';')
    fps_cd = cd_fn(torch.from_numpy(fps1).unsqueeze(0), torch.from_numpy(fps2).unsqueeze(0))
    g_cd = cd_fn(torch.from_numpy(g1).unsqueeze(0), torch.from_numpy(g2).unsqueeze(0))
    print(fps_cd, g_cd)

    #tensor(0.0858, dtype=torch.float64) tensor(0.0224, dtype=torch.float64)