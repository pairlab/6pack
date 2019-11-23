import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from libs.tracker import tracker

# from dataset.dataset_nocs import Dataset
from dataset.eval_dataset_nocs import Dataset
from libs.network import KeyNet
# from libs.loss import Loss

cate_list = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default='dataset', help='dataset root dir')
parser.add_argument('--img_num', type=int, default=500, help='number of training images')
parser.add_argument('--resume', type=str, default='', help='resume model')
parser.add_argument('--num_points', type=int, default=500, help='points')
parser.add_argument('--num_cates', type=int, default = 6, help='number of categories')
parser.add_argument('--workers', type=int, default=5, help='number of data loading workers')
parser.add_argument('--num_kp', type=int, default=8, help='number of kp')
parser.add_argument('--outf', type=str, default='models/', help='save dir')
parser.add_argument('--lr', default=0.0001, help='learning rate')
opt = parser.parse_args()

# model = KeyNet(num_points=opt.num_points, num_key=opt.num_kp, num_cates=opt.num_cates)
# model.cuda()

# dataset = Dataset(opt.dataset_root, opt.num_points, opt.image_num)
# test_dataset = Dataset('val', opt.dataset_root, False, opt.num_points, opt.num_cates, 1000, opt.category)
# testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=opt.workers)

model = KeyNet(num_points = opt.num_points, num_key = opt.num_kp, num_cates = opt.num_cates)
model.cuda()
model.eval()

# for direct forward
choose_cate = 1
choose_obj = "x_hammer1"
choose_video = "x_hammer1"
# choose_video = ""
#                       mode, root,           add_noise, num_pt,       num_cates   count, cate_id
# test_dataset = Dataset('val', opt.dataset_root, False, opt.num_points, choose_cate, 1000)

# for reading from a txt file
# eval_list_file = open('dataset/eval_list/eval_list_{0}.txt'.format(choose_cate), 'r')
# while 1:
#     input_line = eval_list_file.readline()
#     if not input_line:
#         break
#     if input_line[-1:] == '\n':
#         input_line = input_line[:-1]
#     _, choose_obj, choose_video = input_line.split(' ')

# current_r, current_t = test_dataset.getfirst(choose_obj, choose_video)
# print("haha:", current_r, current_t)
# img_fr, choose_fr, cloud_fr, anchor, scale = test_dataset.getone(current_r, current_t)
# img_fr, choose_fr, cloud_fr, anchor, scale = Variable(img_fr).cuda(), \
#                                                  Variable(choose_fr).cuda(), \
#                                                  Variable(cloud_fr).cuda(), \
#                                                  Variable(anchor).cuda(), \
#                                                  Variable(scale).cuda()
# Kp_fr, att_fr = model.eval_forward(img_fr, choose_fr, cloud_fr, anchor, scale, 0.0, True)


###
# num_kp = 8
# num_points = 500
# num_cates = 6
# outf = 'models'
# min_dis = 0.0005
# dataprocess = Dataset(.num_points)
# model = KeyNet(num_points=num_points, num_key=num_kp, num_cates=num_cates)
# model.cuda()
# # model.load_state_dict(torch.load('{0}/{1}'.format(self.outf, self.resume_models[choose_cate - 1])))
# model.eval()
# current_r, current_t = np.eye(3), np.array([0 0 0])
# # self.bbox = [[0.0, 0.0, 0.0] for k in range(8)]
# bbox =
# dataprocess.add_bbox(bbox)
# img_fr, choose_fr, cloud_fr, anchor, scale = self.dataprocess.getone(rgb_dir, depth_dir, current_r, current_t)
# img_fr, choose_fr, cloud_fr, anchor, scale = Variable(img_fr).cuda(), \
#                                              Variable(choose_fr).cuda(), \
#                                              Variable(cloud_fr).cuda(), \
#                                              Variable(anchor).cuda(), \
#                                              Variable(scale).cuda()
# Kp_fr, _ = model.eval_forward(img_fr, choose_fr, cloud_fr, anchor, scale, 0.0, True)

st = 'fr'
ed = 'to'

# category id
category_id = 1
tracker = tracker(category_id)

# provide the initial pose and scale of the object
current_r, current_t = np.eye(3), np.array([0,0,0])
bbox = np.array([[-0.11, -0.11, -0.11], [0.11, 0.11, 0.11], [-0.11, -0.11, -0.11], [0.11, 0.11, 0.11],
                 [-0.11, -0.11, -0.11], [0.11, 0.11, 0.11], [-0.11, -0.11, -0.11], [0.11, 0.11, 0.11]])
# img_dir = 'inference/{0}.png'.format(st)
# depth_dir = 'inference/{0}.png'.format(st)
img_dir = 'inference/pose.png'
depth_dir = 'inference/depth.png'
# current_r, current_t = tracker.init_estimation(img_dir, depth_dir, current_r, current_t, bbox)
kp_Fr = tracker.init_estimation(img_dir, depth_dir, current_r, current_t, bbox)
print(kp_Fr)
