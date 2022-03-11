import torch
import numpy as np
import open3d as o3d
from dataset import SharpNetCompletionDataset
from model import CompletionTransformer
from loss import CDList,  DensityLoss

batch_size = 3
epoch = 100
lr = 0.0003
k = 20
w = 22
cls_ = 4
param_load_path = "../params/completion-step2-w%d-ptconv-k20-cd-adam-upatte.pth" % w
param_save_path = "../params/completion-step2-w%d-ptconv-k20-cd-adam-upatte.pth" % w

#
# param_load_path = "../params/completion-step2-w%d-ptconv-k20-cd-adam-upatte_.pth" % w
# param_save_path = "../params/completion-step2-w%d-ptconv-k20-cd-adam-upatte_.pth" % w


device = torch.device("cuda:0")
net = CompletionTransformer(n_shift_points=7, d=128, n_encoder=6, n_head=8, k=k, downsample="PointConv")
net.to(device)
net.load_state_dict(torch.load(param_load_path))
loss_fn = CDList()
#loss_fn = EMDList()
density_loss = DensityLoss()
optimizer = torch.optim.Adam(lr=lr, params=net.parameters(), weight_decay=0)
train_dataset = SharpNetCompletionDataset(json_path="train_test_split/shuffled_train_file_list.json",w=w, cls_=None)
test_dataset = SharpNetCompletionDataset(json_path="train_test_split/shuffled_test_file_list.json", cls_=None, w=w)



def processbar(current, totle):
    process_str = ""
    for i in range(int(20*current/totle)):
        process_str += "█"
    while len(process_str) < 20:
        process_str += " "
    return "%s|  %d / %d" % (process_str, current, totle)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.eye(num_classes)[y.cpu().data.numpy(), ].to(device)


def get_batch(rand_idx_list, start_idx):
    remain_pc_list, crop_list, crop_grid_list = [], [], []
    inp_batch, need_pts_num, cls_list = [], [], []
    for i in range(start_idx, start_idx+batch_size):
        remain_pc_list_, crop_list_, _, crop_grid_list_, _, _, cls_ = train_dataset[rand_idx_list[i]]
        remain_pc_list, crop_list, crop_grid_list, cls_list = remain_pc_list+remain_pc_list_, crop_list+crop_list_, crop_grid_list+crop_grid_list_, cls_list+cls_
    max_points_num = 0
    for i in range(len(remain_pc_list)):
        max_points_num = max(max_points_num, remain_pc_list[i].shape[0]+crop_grid_list[i].shape[0])
        inp_batch.append(torch.cat([remain_pc_list[i], crop_grid_list[i]], dim=0))
        need_pts_num.append(crop_grid_list[i].shape[0])
    # print(max_points_num)
    for i in range(len(inp_batch)):
        if inp_batch[i].shape[0] < max_points_num:
            pad = inp_batch[i][0].view(1, 3).repeat([max_points_num-inp_batch[i].shape[0], 1])
            inp_batch[i] = torch.cat([pad, inp_batch[i]], dim=0)
        # print(inp_batch[i].shape)
    inp_batch = torch.stack(inp_batch, dim=0)
    categorical = to_categorical(torch.LongTensor(cls_list), 16)
    # print(inp_batch.shape)
    for i in range(len(crop_list)):
        crop_list[i] = crop_list[i].to(device)
    return inp_batch, need_pts_num, crop_list, categorical


if __name__ == '__main__':
    start_idxes = torch.arange(0, len(train_dataset)//batch_size) * batch_size
    def look_look():
        net.eval()
        with torch.no_grad():
            for i in range(len(test_dataset)):
                remain_pc_list_, crop_list_, _, crop_grid_list_, _, _, cls_ = test_dataset[i]
                for j, remain_pc in enumerate(remain_pc_list_):
                    remain_pc = remain_pc.to(device)
                    crop_grid = crop_grid_list_[j].to(device)
                    inp = torch.cat([remain_pc, crop_grid], dim=0).unsqueeze(0)
                    need_num = [crop_grid_list_[j].shape[0]]
                    shifted = net(inp, to_categorical(torch.LongTensor([cls_[j]]), 16), need_num)[0].cpu().numpy()
                    # open3d
                    remain_pts = remain_pc.cpu().numpy()
                    remain_pc = o3d.geometry.PointCloud()
                    remain_pc.points = o3d.Vector3dVector(remain_pts)
                    remain_pc.colors = o3d.Vector3dVector(np.array([[1, 0.706, 0]] * remain_pts.shape[0]))
                    # before completion
                    crop_pts = crop_grid.cpu().numpy()
                    crop_pc = o3d.geometry.PointCloud()
                    crop_pc.points = o3d.Vector3dVector(crop_pts)
                    crop_pc.colors = o3d.Vector3dVector(np.array([[0, 0.651, 0.929]] * crop_pts.shape[0]))
                    # after completion
                    shifted_pts = shifted
                    shifted_pc = o3d.geometry.PointCloud()
                    shifted_pc.points = o3d.Vector3dVector(shifted_pts)
                    shifted_pc.colors = o3d.Vector3dVector(np.array([[0, 0.651, 0.929]] * shifted_pts.shape[0]))

                    o3d.draw_geometries([remain_pc, crop_pc], window_name="test", width=800, height=600)
                    o3d.draw_geometries([remain_pc, shifted_pc], window_name="test", width=800, height=600)

    def train():
        min_loss = 1e8
        for epoch_count in range(1, epoch+1):
            rand_idx_list = np.random.permutation(len(train_dataset))
            net.train()
            process, loss_val = 0, 0
            pred_to_gt, gt_to_pred = 0, 0
            for idx in start_idxes:
                inp_batch, need_pts_num, target, categorical = get_batch(rand_idx_list, idx)
                inp_batch, categorical = inp_batch.to(device), categorical.to(device)
                shifteds = net(inp_batch, categorical, need_pts_num)
                #d_loss = density_loss(shifteds)
                loss = loss_fn(shifteds, target, False)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # test cd
                loss_val += loss.item()
                pred_to_gt_mean, gt_to_pred_mean = loss_fn(shifteds, target, True)
                pred_to_gt += pred_to_gt_mean.item()
                gt_to_pred += gt_to_pred_mean.item()

                process += batch_size
                print("\r测试进度：%s  pred to gt: %.5f  gt to pred: %.5f  " % (
                    processbar(process, len(start_idxes)*batch_size), pred_to_gt_mean.item(), gt_to_pred_mean.item()
                ), end="")
            pred_to_gt, gt_to_pred = pred_to_gt / len(start_idxes), gt_to_pred / len(start_idxes)
            print("\nepoch: %d  pred to gt: %.5f  gt to pred: %.5f" % (epoch_count, pred_to_gt, gt_to_pred))
            if min_loss > loss_val:
                min_loss = loss_val
                print("save...........")
                torch.save(net.state_dict(), param_save_path)
                print("save finished !!!!!!!")
    train()
    # look_look()