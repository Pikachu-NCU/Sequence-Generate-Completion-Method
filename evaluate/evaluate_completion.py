import torch
from model import Transformer, CompletionTransformer
import open3d as o3d
from dataset import SharpNetCompletionDataset
import numpy as np
from loss import CDList
import utils


w = 22
d = 2 / w
k = 20
param1_load_path = "../params/transformer-w%d-pos-emb-mask-fix.pth" % w
param2_load_path = "../params/completion-step2-w%d-ptconv-k20-cd-adam-upatte.pth" % w
gpu, cpu = torch.device("cuda:0"), torch.device("cpu")

voc_size = w**3
pad_num, start_num, end_num = w**3-1, w**3-2, w**3-3
step_1_net = Transformer(inp_voc_size=voc_size, out_voc_size=voc_size, out_dim=voc_size, d=512,
                         n_encoder=6, n_encoder_head=8, n_decoder=6, n_decoder_head=8, pad_num=pad_num)
step_2_net = CompletionTransformer(n_shift_points=7, d=128, n_encoder=6, n_head=8, k=k, downsample="PointConv")
loss_fn = CDList()
# step 1
step_1_net.to(gpu)
step_1_net.load_state_dict(torch.load(param1_load_path))
step_1_net.to(cpu)
# step 2
step_2_net.to(gpu)
step_2_net.load_state_dict(torch.load(param2_load_path))
step_2_net.to(cpu)
print("init finish")


def processbar(current, totle):
    process_str = ""
    for i in range(int(20*current/totle)):
        process_str += "█"
    while len(process_str) < 20:
        process_str += " "
    return "%s|  %d / %d" % (process_str, current, totle)


def to_categorical(y, num_classes):
    return torch.eye(num_classes)[y.cpu().data.numpy(), ].to(gpu)


def evaluate():
    step_1_net.eval()
    # step_2_net.eval()
    cls_list = ["Airplane", "Bag", "Cap", "Car", "Chair", "Earphone",
                "Guitar", "Knife", "Lamp", "Laptop", "Motorbike", "Mug",
                "Pistol", "Rocket", "Skateboard", "Table"]
    utils.fps_rand = False
    # test_cls = None
    with torch.no_grad():
        for test_cls in range(0, 16):
            test_dataset = SharpNetCompletionDataset(json_path="train_test_split/shuffled_test_file_list.json", cls_=None, w=w)
            rand_idx = torch.arange(0, len(test_dataset))

            # print("%s" % cls_list[test_cls])
            process = 0
            pred_to_gt, gt_to_pred = 0, 0

            for i in range(len(test_dataset)):
                step_1_net.to(gpu)

                remain_pc_list, crop_list, remain_grid_list, crop_grid_list, remain_grid_id_list, crop_grid_id_list, cls_ = test_dataset[rand_idx[i]]
                decode_grid_list = []
                for remain_grid_id in remain_grid_id_list:
                    remain_grid_id = torch.LongTensor(remain_grid_id).unsqueeze(0)
                    remain_grid_id = remain_grid_id.to(gpu)
                    cur_num = [start_num]
                    last_num = start_num
                    pts = []
                    pts_num, max_pt_num = 0, 400
                    while last_num != end_num and pts_num < max_pt_num:
                        inp = torch.LongTensor([cur_num]).to(gpu)
                        decoder_out = step_1_net(remain_grid_id, inp)
                        # decoder_out = decoder_out.view(-1, w).argmax(1).view(-1, 3)
                        decoder_out = decoder_out.view(-1, voc_size).argmax(1)
                        # last_xyz = decoder_out[-1, :]
                        last_xyz = [decoder_out[-1] // w ** 2, decoder_out[-1] % w ** 2 // w, decoder_out[-1] % w ** 2 % w]
                        pts.append([last_xyz[0] * d + 0.5 * d, last_xyz[1] * d + 0.5 * d, last_xyz[2] * d + 0.5 * d])
                        # last_num = last_xyz[0]*w**2+last_xyz[1]*w+last_xyz[2]
                        last_num = decoder_out[-1]
                        cur_num.append(last_num)
                        pts_num += 1
                    # print(pts_num)
                    pts = np.array(pts)[:-1, :] - 1
                    decode_grid_list.append(pts)
                # print(len(decode_grid_list))
                process += 1
                step_1_net.to(cpu)
                # step 2
                # 算最多的点数
                max_pt_num = 0
                need_pts_num = []
                for j in range(len(decode_grid_list)):
                    max_pt_num = max(max_pt_num, remain_pc_list[j].shape[0]+decode_grid_list[j].shape[0])
                    need_pts_num.append(decode_grid_list[j].shape[0])
                # 把点补成一样多
                inp = []
                for j in range(len(decode_grid_list)):
                    if remain_pc_list[j].shape[0]+decode_grid_list[j].shape[0] < max_pt_num:
                        need_num = max_pt_num - (remain_pc_list[j].shape[0]+decode_grid_list[j].shape[0])
                        remain_pc_list[j] = torch.cat([remain_pc_list[j][0].view(1, 3).repeat([need_num, 1]), remain_pc_list[j]], dim=0)
                    inp.append(torch.cat([remain_pc_list[j], torch.Tensor(decode_grid_list[j].astype(np.float32))], dim=0))
                step_2_net.to(gpu)
                inp = torch.stack(inp, dim=0).to(gpu)
                categorical = to_categorical(torch.LongTensor(cls_), 16)
                shifteds = step_2_net(inp, categorical, need_pts_num)
                for j in range(len(crop_list)):
                    crop_list[j] = crop_list[j].to(gpu)
                pred_to_gt_mean, gt_to_pred_mean = loss_fn(shifteds, crop_list, True)


                pred_to_gt += pred_to_gt_mean.item()
                gt_to_pred += gt_to_pred_mean.item()
                step_2_net.to(cpu)
                print("\rtest process: %s  pred to gt: %.5f   gt to pred: %.5f" % (processbar(process, len(test_dataset)), pred_to_gt / process, gt_to_pred / process), end="")

                # look look
                for j in range(len(decode_grid_list)):
                    # open3d
                    remain_pts = remain_pc_list[j].cpu().numpy()
                    remain_pc = o3d.geometry.PointCloud()
                    remain_pc.points = o3d.Vector3dVector(remain_pts)
                    remain_pc.colors = o3d.Vector3dVector(np.array([[1, 0.706, 0]] * remain_pts.shape[0]))
                    # before completion
                    crop_pts = decode_grid_list[j]
                    crop_pc = o3d.geometry.PointCloud()
                    crop_pc.points = o3d.Vector3dVector(crop_pts)
                    crop_pc.colors = o3d.Vector3dVector(np.array([[0, 0.651, 0.929]] * crop_pts.shape[0]))
                    # after completion
                    shifted_pts = shifteds[j].cpu().numpy()
                    shifted_pc = o3d.geometry.PointCloud()
                    shifted_pc.points = o3d.Vector3dVector(shifted_pts)
                    shifted_pc.colors = o3d.Vector3dVector(np.array([[0, 0.651, 0.929]] * shifted_pts.shape[0]))

                    # o3d.draw_geometries([remain_pc, crop_pc], window_name="test step1", width=1000, height=800)
                    # o3d.draw_geometries([remain_pc, shifted_pc], window_name="test step2", width=1000, height=800)

            pred_to_gt, gt_to_pred = pred_to_gt / len(test_dataset), gt_to_pred / len(test_dataset)
            print("\n%s  pred to gt: %.5f   gt to pred: %.5f" % (cls_list[test_cls], pred_to_gt, gt_to_pred))
    utils.fps_rand = True


if __name__ == '__main__':
    evaluate()