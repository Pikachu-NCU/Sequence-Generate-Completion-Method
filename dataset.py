import os
import json
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from utils import pc_normalize, farthest_point_sample, pc_struct, point_cloud_read
import torch
import open3d as o3d


class PartNormalDataset(Dataset):
    def __init__(self, root='zhang/dataset/shapenetcore_partanno_segmentation_benchmark_v0_normal', npoints=2500,
                 split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.normal_channel = normal_channel

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        self.cat = {k: v for k, v in self.cat.items()}
        self.classes_original = dict(zip(self.cat, range(len(self.cat))))

        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}
        # print(self.cat)

        self.meta = {}
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
            train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
            val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
            test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
        for item in self.cat:
            # print('category', item)
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item])
            fns = sorted(os.listdir(dir_point))
            # print(fns[0][0:-4])
            if split == 'trainval':
                fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
            elif split == 'train':
                fns = [fn for fn in fns if fn[0:-4] in train_ids]
            elif split == 'val':
                fns = [fn for fn in fns if fn[0:-4] in val_ids]
            elif split == 'test':
                fns = [fn for fn in fns if fn[0:-4] in test_ids]
            else:
                print('Unknown split: %s. Exiting..' % (split))
                exit(-1)

            # print(os.path.basename(fns))
            for fn in fns:
                token = (os.path.splitext(os.path.basename(fn))[0])
                self.meta[item].append(os.path.join(dir_point, token + '.txt'))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn))

        self.classes = {}
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

        # for cat in sorted(self.seg_classes.keys()):
        #     print(cat, self.seg_classes[cat])

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 20000

        self.split = split

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls, seg = self.cache[index]
        else:
            fn = self.datapath[index]
            cat = self.datapath[index][0]
            cls = self.classes[cat]
            cls = np.array([cls]).astype(np.int32)
            data = np.loadtxt(fn[1]).astype(np.float32)
            if not self.normal_channel:
                point_set = data[:, 0:3]
            else:
                point_set = data[:, 0:6]
            seg = data[:, -1].astype(np.int32)
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls, seg)
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        # if self.split != "test":
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        seg = seg[choice]

        return point_set, cls, seg

    def __len__(self):
        return len(self.datapath)


class ModelNet40Dataset(Dataset):
    def __init__(self, root,  npoint=1024, split='train', uniform=False, normal_channel=True, cache_size=15000):
        self.root = root
        self.npoints = npoint
        self.uniform = uniform
        self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
        shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        # list of (shape_name, shape_txt_file_path) tuple
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d'%(split,len(self.datapath)))

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]

            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

            if not self.normal_channel:
                point_set = point_set[:, 0:3]

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        return point_set, cls

    def __getitem__(self, index):
        return self._get_item(index)


class SharpNetCompletionDataset(Dataset):
    def __init__(self, root="../shapenetcore_partanno_segmentation_benchmark_v0/",
                 json_path="train_test_split/shuffled_train_file_list.json", cls_=None, get_label=False,
                 w=30, cache_size=15000):
        super(SharpNetCompletionDataset, self).__init__()
        with open(root+json_path, "r") as f:
            files = json.load(f)
        code2label = {
            "02691156": 0,  "02773838": 1,  "02954340": 2,  "02958343": 3,
            "03001627": 4,  "03261776": 5,  "03467517": 6,  "03624134": 7,
            "03636649": 8,  "03642806": 9,  "03790512": 10, "03797390": 11,
            "03948459": 12, "04099429": 13, "04225987": 14, "04379243": 15
        }
        self.get_label = get_label
        self.files = []
        for file in files:
            code, path = file[11:19], file[20:]
            if cls_ is None:
                self.files.append([root+code+"/points/"+path+".pts", code2label[code]])
            elif code2label[code] == cls_:
                self.files.append([root + code + "/points/"+path+".pts", code2label[code]])
        self.w = w

        self.cache_size = cache_size
        self.cache = {}
        #self.files = self.files[:4]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        if index in self.cache:
            pts, label = self.cache[index]
        else:
            path, label = self.files[index]
            pts = pc_normalize(point_cloud_read(path))
        if len(self.cache) < self.cache_size:
            self.cache[index] = (pts, label)

        pts = torch.Tensor(pts)
        remain_pc_list, crop_list, remain_grid_list, crop_grid_list = [], [], [], []
        remain_grid_id_list, crop_grid_id_list = [], []
        viewpoints = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]), torch.Tensor([-1, 0, 0])]
        drop_num = pts.shape[0] // 4
        for viewpoint in viewpoints:
            # viewpoint = viewpoint.to(device)
            crop_idx = torch.topk(((pts-viewpoint.unsqueeze(0))**2).sum(1), dim=0, k=drop_num, largest=False)[1]
            remain_idx = torch.topk(((pts-viewpoint.unsqueeze(0))**2).sum(1), dim=0, k=pts.shape[0]-drop_num, largest=True)[1]
            crop_cloud, remain_cloud = pts[crop_idx], pts[remain_idx]

            remain_pc_list.append(remain_cloud)
            crop_list.append(crop_cloud)

            remain_cloud_grid, remain_grid_id = pc_struct(remain_cloud.cpu().numpy(), w=self.w)
            crop_cloud_grid, crop_grid_id = pc_struct(crop_cloud.cpu().numpy(), w=self.w)

            remain_cloud_grid, remain_grid_id = torch.Tensor(remain_cloud_grid), torch.LongTensor(remain_grid_id)
            crop_cloud_grid, crop_grid_id = torch.Tensor(crop_cloud_grid), torch.LongTensor(crop_grid_id)

            remain_grid_list.append(remain_cloud_grid)
            crop_grid_list.append(crop_cloud_grid)
            remain_grid_id_list.append(remain_grid_id)
            crop_grid_id_list.append(crop_grid_id)

            # pc_remain = o3d.geometry.PointCloud()
            # pc_remain.points = o3d.cpu.pybind.utility.Vector3dVector(remain_cloud)
            # print(remain_cloud.shape)
            # pc_remain.colors = o3d.cpu.pybind.utility.Vector3dVector(
            #     np.array([[1, 0.706, 0]] * remain_cloud.shape[0]))
            # pc_crop = o3d.geometry.PointCloud()
            # pc_crop.points = o3d.cpu.pybind.utility.Vector3dVector(crop_cloud_grid)
            # pc_crop.colors = o3d.cpu.pybind.utility.Vector3dVector(np.array([[0, 0.651, 0.929]] * crop_cloud_grid.shape[0]))
            # o3d.visualization.draw_geometries([pc_remain, pc_crop], window_name="test", width=800, height=600)

        return remain_pc_list, crop_list, remain_grid_list, crop_grid_list, remain_grid_id_list, crop_grid_id_list, [label]*len(viewpoints)


if __name__ == '__main__':
    completion_dataset = SharpNetCompletionDataset(w=30)
    for i in range(len(completion_dataset)):
        _, _, _, _, _, _ = completion_dataset[i]