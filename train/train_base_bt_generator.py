import torch
import numpy as np
from dataset import SharpNetCompletionDataset
from model import Transformer
import open3d as o3d
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR


# 真正的 batch_size 是 batch_size x 4, 4是视点的数量
batch_size = 3
epoch = 200
lr = 0.01
min_lr = 0.00001
lr_update_step = 20
lr_update_gamma = 0.5
# 每条边分成多少个格子
w = 22
patch_size = 6
param_load_path = "../params/transformer-w%d-pos-emb-mask-fix.pth" % w
param_save_path = "../params/transformer-w%d-pos-emb-mask-fix.pth" % w
cls_ = None
train_dataset = SharpNetCompletionDataset(json_path="train_test_split/shuffled_train_val_file_list.json", cls_=cls_, w=w)
test_dataset = SharpNetCompletionDataset(json_path="train_test_split/shuffled_test_file_list.json", cls_=cls_, w=w)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# scheduler = StepLR(optimizer, 10, gamma=0.7)
voc_size = w**3
pad_num, start_num, end_num = w**3-1, w**3-2, w**3-3
net = Transformer(inp_voc_size=voc_size, out_voc_size=voc_size, out_dim=voc_size, d=512,
                  n_encoder=6, n_encoder_head=8, n_decoder=6, n_decoder_head=8, pad_num=pad_num)
net.to(device)
net.load_state_dict(torch.load(param_load_path))
optimizer = torch.optim.SGD(lr=lr, params=net.parameters(), momentum=0.9)
# beta1, beta2 = 0.9, 0.98
# optimizer = torch.optim.Adam(lr=lr, params=net.parameters(), betas=(beta1, beta2), eps=1e-9, weight_decay=0)
# optimizer = torch.optim.AdamW(lr=lr, params=net.parameters())
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_num)


def processbar(current, totle):
    process_str = ""
    for i in range(int(20*current/totle)):
        process_str += "█"
    while len(process_str) < 20:
        process_str += " "
    return "%s|  %d / %d" % (process_str, current, totle)


def update_lr(optimizer, gamma=0.5):
    global lr
    lr = max(lr*gamma, min_lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print("lr update finished  cur lr: %.5f" % lr)


def get_batch(data_set, rand_idx, start_i, batch_size, pad_num, start_num, end_num):
    max_remain_pt_num, max_crop_pt_num = 0, 0
    remain_pc_grid_id_list, crop_pc_grid_id_list = [], []
    crop_pc_grid_inp_id, crop_pc_grid_label_id = [], []
    for idx in range(start_i, start_i + batch_size):
        dataset_idx = rand_idx[idx]
        remain_pc_list, crop_list, remain_grid_list, crop_grid_list, remain_grid_id_list, crop_grid_id_list, _ = data_set[dataset_idx]
        # 算出最多的点数
        for i, remain_pc_grid in enumerate(remain_grid_list):
            if max_remain_pt_num < remain_pc_grid.shape[0]:
                max_remain_pt_num = remain_pc_grid.shape[0]
            if max_crop_pt_num < crop_grid_list[i].shape[0]:
                max_crop_pt_num = crop_grid_list[i].shape[0]
        remain_pc_grid_id_list = remain_pc_grid_id_list + remain_grid_id_list
        crop_pc_grid_id_list = crop_pc_grid_id_list + crop_grid_id_list
    for i in range(len(remain_pc_grid_id_list)):
        # 给输入的点的编号补上padding，因为点云的数量不一致
        remain_pc_grid_id_list[i] = np.concatenate([remain_pc_grid_id_list[i], np.array([pad_num]*(max_remain_pt_num-remain_pc_grid_id_list[i].shape[0]))], axis=0)
        # 每个输入前补上start_num
        inp = np.concatenate([np.array([start_num]), crop_pc_grid_id_list[i]], axis=0)
        # 给输入补padding
        inp = np.concatenate([inp, np.array([pad_num]*(max_crop_pt_num+1-inp.shape[0]))], axis=0)
        crop_pc_grid_inp_id.append(inp)
        # 每个输出最后补上end_num
        out = np.concatenate([crop_pc_grid_id_list[i], np.array([end_num])], axis=0)
        # 给输出补padding
        out = np.concatenate([out, np.array([pad_num]*(max_crop_pt_num+1-out.shape[0]))], axis=0)
        crop_pc_grid_label_id.append(out)
        # print(remain_pc_grid_id_list[i].shape[0], inp.shape[0], out.shape[0])
    remain_pc_grid_id_list, crop_pc_grid_inp_id, crop_pc_grid_label_id = np.stack(remain_pc_grid_id_list, axis=0), np.stack(crop_pc_grid_inp_id, axis=0), np.stack(crop_pc_grid_label_id, axis=0)
    remain_pc_grid_id_list, crop_pc_grid_inp_id, crop_pc_grid_label_id = remain_pc_grid_id_list.astype(np.int), crop_pc_grid_inp_id.astype(np.int), crop_pc_grid_label_id.astype(np.int)
    # print(remain_pc_grid_id_list.shape, crop_pc_grid_inp_id.shape, crop_pc_grid_label_id.shape)
    rand_index = torch.LongTensor(np.random.permutation(remain_pc_grid_id_list.shape[0]))
    remain_pc_grid_id_list = remain_pc_grid_id_list[rand_index]
    crop_pc_grid_inp_id = crop_pc_grid_inp_id[rand_index]
    crop_pc_grid_label_id = crop_pc_grid_label_id[rand_index]
    # 该返回方式直接返回label，label形状为batch_size x n
    return torch.LongTensor(remain_pc_grid_id_list), torch.LongTensor(crop_pc_grid_inp_id), torch.LongTensor(crop_pc_grid_label_id)


def evaluate():
    net.eval()
    rand_idx = np.random.permutation(len(test_dataset))
    process, loss_val, correct_num, totle_num = 0, 0, 0, 0
    with torch.no_grad():
        for i in range(0, len(test_dataset), batch_size):
            if len(test_dataset) - i < batch_size:
                continue
            encoder_inp, decoder_inp, decoder_label = get_batch(test_dataset, rand_idx, i, batch_size, pad_num, start_num, end_num)
            encoder_inp, decoder_inp, decoder_label = encoder_inp.to(device), decoder_inp.to(device), decoder_label.to(device)
            # print(encoder_inp.shape, decoder_inp.shape, decoder_label.shape)
            decoder_out = net(encoder_inp, decoder_inp)
            # decode_out:    batch_size x m x voc_size
            # decoder_label: batch_size x m
            # pad_num is not create loss
            decoder_label = decoder_label.view(-1)
            valid_index = (decoder_label != pad_num)

            process += batch_size
            correct_num += (decoder_out.view(-1, voc_size).argmax(1)[valid_index] == decoder_label[valid_index]).sum(dim=0).item()
            totle_num += valid_index.shape[0]
            print("\rtest process: %s  acc: %.5f" % (processbar(process, len(test_dataset)), correct_num / totle_num), end="")
            # torch.cuda.empty_cache()
        acc = correct_num / totle_num
        print("\ntest finished !!!  acc: %.5f" % acc)
    return acc


def train():
    max_acc = 0
    for epoch_count in range(1, epoch+1):
        net.train()
        rand_idx = np.random.permutation(len(train_dataset))
        process, loss_val, correct_num, totle_num = 0, 0, 0, 0
        for i in range(0, len(train_dataset), batch_size):
            if len(train_dataset) - i < batch_size:
                continue
            encoder_inp, decoder_inp, decoder_label = get_batch(train_dataset, rand_idx, i, batch_size, pad_num, start_num, end_num)
            encoder_inp, decoder_inp, decoder_label = encoder_inp.to(device), decoder_inp.to(device), decoder_label.to(device)
            # print(encoder_inp.shape, decoder_inp.shape, decoder_label.shape)
            decoder_out = []
            for j in range(encoder_inp.shape[0] // patch_size):
                patch_en_inp, patch_de_inp, patch_label = encoder_inp[j*patch_size:(j+1)*patch_size, :], decoder_inp[j*patch_size:(j+1)*patch_size, :], decoder_label[j*patch_size:(j+1)*patch_size, :]
                patch_out = net(patch_en_inp, patch_de_inp)


                # decode_out:    batch_size x m x voc_size
                # decoder_label: batch_size x m
                # pad_num is not create loss
                decoder_out.append(patch_out)
                loss = loss_fn(patch_out.view(-1, voc_size), patch_label.view(-1))
                # loss = loss_fn(decoder_out.view(-1, voc_size)[valid_index], decoder_label[valid_index])
                loss_val += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            decoder_out = torch.cat(decoder_out, dim=0)
            valid_index = (decoder_label.view(-1) != pad_num)
            process += batch_size
            # print(decoder_out.shape, decoder_label.shape)
            correct_num += (decoder_out.view(-1, voc_size).argmax(1)[valid_index] == decoder_label.view(-1)[valid_index]).sum(dim=0).item()
            totle_num += valid_index.shape[0]

            print("\rprocess: %s  loss: %.5f  acc: %.5f" % (processbar(process, len(train_dataset)), loss.item(), correct_num / totle_num), end="")
            del loss
            torch.cuda.empty_cache()
        print("\nepoch: %d  loss: %.5f  acc: %.5f" % (epoch_count, loss_val, correct_num / totle_num))
        acc = evaluate()
        if max_acc < acc:
            max_acc = acc
            print("save...")
            torch.save(net.state_dict(), param_save_path)
            print("finish !!!")
        if epoch_count % lr_update_step == 0:
            update_lr(optimizer, lr_update_gamma)


def look_look():
    order_idx = np.arange(0, len(test_dataset))
    rand_idx = np.random.permutation(len(test_dataset))
    d = 2 / w
    net.eval()
    with torch.no_grad():
        for i in range(len(test_dataset)):
            points, _, remain_grid_list, crop_grid_list, remain_grid_id_list, crop_grid_id_list, cls_ = test_dataset[i]
            #_, _, remain_grid_list, crop_grid_list, remain_grid_id_list, crop_grid_id_list, cls_ = test_dataset[rand_idx[i]]
            for i, remain_grid_id in enumerate(remain_grid_id_list):
                remain_grid_id = torch.LongTensor(remain_grid_id).unsqueeze(0)
                remain_grid_id = remain_grid_id.to(device)
                cur_num = [start_num]
                last_num = start_num
                pts = []
                pts_num, max_pt_num = 0, 400
                #
                pointss = np.array(points[i])
                #
                while last_num != end_num and pts_num < max_pt_num:
                    inp = torch.LongTensor([cur_num]).to(device)
                    decoder_out = net(remain_grid_id, inp)
                    # decoder_out = decoder_out.view(-1, w).argmax(1).view(-1, 3)
                    decoder_out = decoder_out.view(-1, voc_size).argmax(1)
                    # last_xyz = decoder_out[-1, :]
                    last_xyz = [decoder_out[-1] // w ** 2, decoder_out[-1] % w ** 2 // w, decoder_out[-1] % w ** 2 % w]
                    pts.append([last_xyz[0] * d + 0.5 * d, last_xyz[1] * d + 0.5 * d, last_xyz[2] * d + 0.5 * d])
                    # last_num = last_xyz[0]*w**2+last_xyz[1]*w+last_xyz[2]
                    last_num = decoder_out[-1]
                    cur_num.append(last_num)
                    pts_num += 1
                print(pts_num)
                pts = np.array(pts)[:-1, :] - 1
                pc_remain = o3d.PointCloud()
                pc_remain.points = o3d.Vector3dVector(remain_grid_list[i])
                pc_remain.colors = o3d.Vector3dVector(np.array([[1, 0.706, 0]] * remain_grid_list[i].shape[0]))
                pc_crop = o3d.geometry.PointCloud()
                pc_crop.points = o3d.Vector3dVector(pts)
                pc_crop.colors = o3d.Vector3dVector(np.array([[0, 0.651, 0.929]] * pts.shape[0]))
                o3d.draw_geometries([pc_crop, pc_remain], window_name="test", width=1000, height=800)
                # break
            # break


if __name__ == '__main__':
    train()
    # evaluate()
    # look_look()