import torch
from torch import nn
from dataset import ModelNet40Dataset
from model import ClsTransformer
from torch.utils import data
import utils
from utils import random_scale_point_cloud, shift_point_cloud, random_point_dropout, translate_pointcloud
import numpy as np


gpu = torch.device("cuda:0")
k = 20
net = ClsTransformer(num_classes=40, d=128, n_encoder=4, n_head=8, k=k, downsample="PointConv")
net.to(gpu)

param_save_path = "../params/cls-ptconv-k20-crossentropy.pth"
# net.load_state_dict(torch.load(param_save_path))
point_num = 1024
batch_size = 27
lr = 0.001
min_lr = 0.00001
epoch = 300
lr_update_step = 20
lr_update_gamma = 0.5
weight_decay = 1e-4

optimizer = torch.optim.SGD(lr=lr, params=net.parameters(), momentum=0.9, weight_decay=weight_decay)
# optimizer = torch.optim.Adam(lr=lr, params=net.parameters(), weight_decay=0)
# optimizer = torch.optim.AdamW(lr=lr, params=net.parameters(), weight_decay=weight_decay)
loss_fn = nn.CrossEntropyLoss()
dataset_path = "D:\\dataset\\modelnet40_normal_resampled"
modelnet40_train, modelnet40_test = ModelNet40Dataset(root=dataset_path, npoint=point_num, split="train", normal_channel=True), ModelNet40Dataset(root=dataset_path, npoint=point_num, split="test", normal_channel=True)
train_loader = data.DataLoader(modelnet40_train, shuffle=True, batch_size=batch_size)
test_loader = data.DataLoader(modelnet40_test, shuffle=True, batch_size=batch_size)


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


if __name__ == '__main__':
    max_acc = 0


    def evaluate():
        utils.fps_rand = False
        net.eval()
        mean_correct = []
        class_acc = np.zeros((40, 3))
        process, loss_val, tot_correct = 0, 0, 0
        with torch.no_grad():
            for j, data in enumerate(test_loader):
                points, target = data
                target = target[:, 0]
                points, target = points.to(gpu), target.to(gpu)
                pred = net(points)
                pred_choice = pred.data.max(1)[1]
                for cat in np.unique(target.cpu()):
                    classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                    class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
                    class_acc[cat, 1] += 1
                correct = pred_choice.eq(target.long().data).cpu().sum()
                tot_correct += correct.item()
                mean_correct.append(correct.item() / float(points.size()[0]))
                process += points.shape[0]
                print("\r测试进度：%s  当前精度: %.5f" % (processbar(process, len(modelnet40_test)), tot_correct / process), end="")
        class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
        class_acc = np.mean(class_acc[:, 2]).item()
        instance_acc = np.mean(mean_correct).item()
        utils.fps_rand = True
        print("\n测试完毕  ins acc: %.5f  cls acc: %.5f  max acc: %.5f" % (instance_acc, class_acc, max_acc))
        return instance_acc, class_acc


    def train():
        global max_acc
        for epoch_count in range(1, epoch + 1):
            net.train()
            loss_val, process, correct = 0, 0, 0
            for pts, label in train_loader:
                pts = pts.data.numpy()
                pts = random_point_dropout(pts)
                # pts = translate_pointcloud(pts)
                pts[:, :, 0:3] = random_scale_point_cloud(pts[:, :, 0:3])
                pts[:, :, 0:3] = shift_point_cloud(pts[:, :, 0:3])
                pts = torch.Tensor(pts)
                pts, label = pts.float().to(gpu), label.long().view(-1).to(gpu)
                out = net(pts)


                correct += (out.argmax(dim=1) == label).sum(dim=0).item()
                loss = loss_fn(out, label.to(gpu))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_val += loss.item()
                process += pts.shape[0]

                print("\r进度：%s  本批loss:%.5f  当前精度: %.5f" % (processbar(process, len(modelnet40_train)), loss.item(), correct / process), end="")
            print("\nepoch:%d  loss:%.3f" % (epoch_count, loss_val))

            print("开始测试...")
            ins_acc, cls_acc = evaluate()
            if max_acc < ins_acc:
                max_acc = ins_acc
                print("save...")
                torch.save(net.state_dict(), param_save_path)
                print("save finished !!!")
            if epoch_count % lr_update_step == 0:
                update_lr(optimizer, lr_update_gamma)
    train()
    # evaluate()