import torch
from dataset import PartNormalDataset
from model import PartSegTransformer
from torch.utils import data
from torch import nn
import numpy as np
import utils
from utils import random_scale_point_cloud, shift_point_cloud
from loss import FocalLoss


device = torch.device("cuda:0")
epoch = 251
point_num = 2048
lr = 0.001
min_lr = 0.00001
lr_update_step = 20
batch_size = 15
# l2 reg only for SGD
weight_decay = 1e-4
param_load_path = "../params/partseg-ptconv-k20-focal-adam-upatte.pth"
param_save_path = "../params/partseg-ptconv-k20-focal-adam-upatte.pth"
# loss_fn = nn.CrossEntropyLoss()
loss_fn = FocalLoss(gamma=2, alpha=1)
k = 20
net = PartSegTransformer(num_classes=50, d=128, n_encoder=6, n_head=8, k=k, downsample="PointConv")
net.to(device)
net.load_state_dict(torch.load(param_load_path))
# optimizer = torch.optim.SGD(params=net.parameters(), lr=lr, weight_decay=weight_decay)
optimizer = torch.optim.Adam(params=net.parameters(), lr=lr, weight_decay=0)
seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
               'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
               'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
               'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
partseg_dataset_train = PartNormalDataset("C:/Users/sdnyz/PycharmProjects/dataset/shapenet_benchmark_v0_normal/",
                                          npoints=point_num, split='trainval', normal_channel=True)
partseg_dataset_test = PartNormalDataset("C:/Users/sdnyz/PycharmProjects/dataset/shapenet_benchmark_v0_normal/",
                                          npoints=point_num, split='test', normal_channel=True)
train_loader = data.DataLoader(dataset=partseg_dataset_train, batch_size=batch_size, shuffle=True)
# test dataset是不能随机取2048个点的，原本有几个点就用几个测，因此batch_size只能是1
test_loader = data.DataLoader(dataset=partseg_dataset_test, batch_size=1, shuffle=False)

print(len(partseg_dataset_train))


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


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.eye(num_classes)[y.cpu().data.numpy(), ].to(device)


if __name__ == '__main__':
    def evaluate():
        # 测试时先把fps的随机性去掉
        utils.fps_rand = False
        net.eval()
        loss_val, process, correct, process_pts = 0, 0, 0, 0
        # 指标统计量
        num_part = 50
        total_seen_class = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        with torch.no_grad():
            for pts, cls_, label in test_loader:
                pts, cls_, label = pts.float().to(device), cls_.long().to(device), label.long().to(device)
                point_num = pts.shape[1]
                out = net(pts, to_categorical(cls_, 16))

                print(out.shape)

                out = out.view(pts.shape[0], -1, num_part)
                cur_pred_val = out.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((pts.shape[0], point_num)).astype(np.int32)
                target = label.cpu().data.numpy()
                for i in range(pts.shape[0]):
                    cat = seg_label_to_cat[target[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                correct += (out.view(-1, num_part).argmax(dim=1) == label.view(-1)).sum(dim=0).item()
                loss = loss_fn(out.view(-1, num_part), label.view(-1))
                loss_val += loss.item()
                process += pts.shape[0]
                process_pts += point_num * pts.shape[0]

                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

                for i in range(pts.shape[0]):
                    segp = cur_pred_val[i, :]
                    segl = target[i, :]
                    cat = seg_label_to_cat[segl[0]]
                    part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                    for l in seg_classes[cat]:
                        if (np.sum(segl == l) == 0) and (np.sum(segp == l) == 0):  # part is not present, no prediction as well
                            part_ious[l - seg_classes[cat][0]] = 1.0
                        else:
                            part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(np.sum((segl == l) | (segp == l)))
                    shape_ious[cat].append(np.mean(part_ious))

                print("\r测试进度：%s  本批loss:%.5f  当前精度: %.5f" % (processbar(process, len(partseg_dataset_test)), loss.item(), correct / process_pts), end="")
        all_shape_ious = []
        for cat in shape_ious.keys():
            for iou in shape_ious[cat]:
                all_shape_ious.append(iou)
            shape_ious[cat] = np.mean(shape_ious[cat])
        mean_shape_ious = np.mean(list(shape_ious.values()))
        # 所有指标汇总统计
        accuracy = correct / process_pts
        class_avg_accuracy = np.mean(np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
        class_avg_iou = mean_shape_ious.item()
        instance_avg_iou = np.mean(all_shape_ious).item()
        print("\ncls iou: ")
        print(shape_ious)
        print("test finished!  accuracy: %.5f   class avg iou: %.5f  instance avg iou: %.5f" % (accuracy, class_avg_iou, instance_avg_iou))

        utils.fps_rand = True
        return accuracy, class_avg_iou

    def train():
        max_iou = 0.82
        for epoch_count in range(1, epoch + 1):
            net.train()
            loss_val, process, correct = 0, 0, 0
            for pts, cls_, label in train_loader:
                pts = pts.data.numpy()

                pts[:, :, 0:3] = random_scale_point_cloud(pts[:, :, 0:3])
                pts[:, :, 0:3] = shift_point_cloud(pts[:, :, 0:3])
                pts = torch.Tensor(pts)
                pts, cls_, label = pts.float().to(device), cls_.long().to(device), label.long().view(-1).to(device)
                # 自己所属的类别也算一个特征，比如自己是飞机啊，汽车啊这种
                out = net(pts, to_categorical(cls_, 16))


                correct += (out.argmax(dim=1) == label).sum(dim=0).item()
                loss = loss_fn(out, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val += loss.item()
                process += pts.shape[0]

                print("\r进度：%s  本批loss:%.5f  当前精度: %.5f" % (processbar(process, len(partseg_dataset_train)), loss.item(), correct / (process*point_num)), end="")
            print("\nepoch:%d  loss:%.3f" % (epoch_count, loss_val))
            print("开始测试...")
            accuracy, miou = evaluate()
            if max_iou < miou:
                max_iou = miou
                print("save...")
                torch.save(net.state_dict(), param_save_path)
                print("save finished !!!")
            if epoch_count % lr_update_step == 0:
                update_lr(optimizer, 0.5)

    # train()
    evaluate()