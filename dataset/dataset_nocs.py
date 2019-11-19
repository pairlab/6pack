import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
from libs.transformations import euler_matrix
import argparse
import time
import random
import numpy.ma as ma
import copy
import math
import scipy.misc
import scipy.io as scio
import cv2

class Dataset(data.Dataset):
    def __init__(self, mode, root, add_noise, num_pt, num_cates, count, cate_id):
        # root directory
        self.root = root
        # add noise
        self.add_noise = add_noise
        # train
        self.mode = mode
        # number of what points?
        self.num_pt = num_pt
        # how many tool shapes
        self.num_cates = num_cates
        # data_augment
        self.back_root = '{0}/train2017/'.format(self.root)
        # load files for training
        # tool shape id
        self.cate_id = cate_id
        # dict { obj id -> obj? }
        self.obj_list = {}
        # dict { obj id -> name? }
        self.obj_name_list = {}

        # load obj_list {cate_id:[item1, item2, item3]}
        # one item can have multiple frames, for our case, just 2 frames, from and to
        # do we need to sample from the list? probably not -> no need to construct this list
        # in get_item, just go through all the data we have
        # do we need small delta of r and t for each step? no, just two frames, no tracking
        # what's syn_or_real? we have syn, not sure about real
        # load obj_name_list {cate_id:{item1:[file path 1, fp2], 2:[fp1, fp2]}}
        if self.mode == 'train':
            # {0} = "Mydata"
            # load tool names for each cate_id
            # eg. cate_id = a, obj_name_list {a:[1, 2, 3]}
            # in our case
            # obj_name_list = {1: ["X_shape_hammer", "L_shape_hammer", "T_shape_hammer"],
            #                       2: [""]}
            # obj_list = {1:{"X_shape_hammer"：["X_shape_hammer_1", "X_shape_hammer_2", "X_shape_hammer_3"],
            #                  2:{"L_shape_hammer"：["L_shape_hammer_1", "L_shape_hammer_2", "L_shape_hammer_3"]}
            self.obj_name_list[cate_id] = os.listdir('{0}/{1}/'.format(self.root, cate_id))
            self.obj_list[cate_id] = {}
            # for tool 1 ,2 ,3 in tool shape a
            for item in self.obj_name_list[cate_id]:
                # tool shape a, tool 1
                print(cate_id, item)
                # {a:{1:[file path 1, file path2], 2:[file path1, file path2]}}
                self.obj_list[cate_id][item] = []
                # read from MyNOCS / data_list / train / temp_cate_id / list.txt
                # My_NOCS\data_list\train\1\x_hammer1
                input_file = open('{0}/{1}/{2}/list.txt'.format(self.root, cate_id, item), 'r')
                while 1:
                    input_line = input_file.readline()
                    if not input_line:
                        break
                    if input_line[-1:] == '\n':
                        input_line = input_line[:-1]
                    # to put image of item in different frames in data folder as described in list.txt
                    self.obj_list[cate_id][item].append('{0}/data/{1}'.format(self.root, input_line))
                # item is tools of different shapes
                # input_file = open('{0}/{1}/{2}/list.txt'.format(self.root, cate_id, item), 'r')
                # self.obj_list[cate_id][item].append('{0}/data/{1}'.format(self.root, input_line))
                input_file.close()
        self.real_obj_list = {}
        self.real_obj_name_list = {}
        # cate_id = 1, 2, ..., num_cates
        for cate_id in range(1, self.num_cates+1):
            # {a: [path to file1, path to file2]}
            # path is MyNOCS/data_list/real_train/a
            # build a training set of
            # cate_id = 1 (bottle), [bottle_blue_google_norm, bottle_starbuck_norm]
            self.real_obj_name_list[cate_id] = os.listdir(
                '{0}/data_list/real_{1}/{2}/'.format(self.root, self.mode, cate_id))
            # real_obj_list
            # {a:{1:[My_NOCS/data/]},
            self.real_obj_list[cate_id] = {}
            for item in self.real_obj_name_list[cate_id]:
                print(cate_id, item)
                self.real_obj_list[cate_id][item] = []
                # path = MyNOCS/ data_list/real_train/ a/1/list.txt
                # list contain images of the same item in different frames
                input_file = open(
                    '{0}/data_list/real_{1}/{2}/{3}/list.txt'.format(self.root, self.mode, cate_id, item), 'r')
                while 1:
                    input_line = input_file.readline()
                    if not input_line:
                        break
                    if input_line[-1:] == '\n':
                        input_line = input_line[:-1]
                self.real_obj_list[cate_id][item].append('{0}/data/{1}'.format(self.root, input_line))
            # in our case
            # real_obj_name_list = {1: ["X_shape_hammer", "L_shape_hammer", "T_shape_hammer"],
            #                       2: [""]}
            # real_obj_list = {1:{"X_shape_hammer"：["X_shape_hammer_1", "X_shape_hammer_2", "X_shape_hammer_3"],
            #                  2:{"L_shape_hammer"：["L_shape_hammer_1", "L_shape_hammer_2", "L_shape_hammer_3"]}

        self.back_list = []

        input_file = open('dataset/train2017.txt', 'r')
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            self.back_list.append(self.back_root + input_line)
        input_file.close()


        self.mesh = []
        input_file = open('dataset/sphere.xyz', 'r')
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            input_line = input_line.split(' ')
            self.mesh.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        input_file.close()
        self.mesh = np.array(self.mesh) * 0.6

        self.cam_cx_1 = 322.52500
        self.cam_cy_1 = 244.11084
        self.cam_fx_1 = 591.01250
        self.cam_fy_1 = 590.16775

        self.cam_cx_2 = 319.5
        self.cam_cy_2 = 239.5
        self.cam_fx_2 = 577.5
        self.cam_fy_2 = 577.5

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.trancolor = transforms.ColorJitter(0.8, 0.5, 0.5, 0.05)
        self.length = count

    def divide_scale(self, scale, pts):
        pts[:, 0] = pts[:, 0] / scale[0]
        pts[:, 1] = pts[:, 1] / scale[1]
        pts[:, 2] = pts[:, 2] / scale[2]

        return pts

    def get_anchor_box(self, ori_bbox):
        bbox = ori_bbox
        limit = np.array(search_fit(bbox))
        num_per_axis = 5
        gap_max = num_per_axis - 1

        small_range = [1, 3]

        gap_x = (limit[1] - limit[0]) / float(gap_max)
        gap_y = (limit[3] - limit[2]) / float(gap_max)
        gap_z = (limit[5] - limit[4]) / float(gap_max)

        ans = []
        scale = [max(limit[1], -limit[0]), max(limit[3], -limit[2]), max(limit[5], -limit[4])]

        for i in range(0, num_per_axis):
            for j in range(0, num_per_axis):
                for k in range(0, num_per_axis):
                    ans.append([limit[0] + i * gap_x, limit[2] + j * gap_y, limit[4] + k * gap_z])

        ans = np.array(ans)
        scale = np.array(scale)

        ans = self.divide_scale(scale, ans)

        return ans, scale

    def change_to_scale(self, scale, cloud_fr, cloud_to):
        cloud_fr = self.divide_scale(scale, cloud_fr)
        cloud_to = self.divide_scale(scale, cloud_to)

        return cloud_fr, cloud_to


    def enlarge_bbox(self, target):

        limit = np.array(search_fit(target))
        longest = max(limit[1]-limit[0], limit[3]-limit[2], limit[5]-limit[4])
        longest = longest * 1.3

        scale1 = longest / (limit[1]-limit[0])
        scale2 = longest / (limit[3]-limit[2])
        scale3 = longest / (limit[5]-limit[4])

        target[:, 0] *= scale1
        target[:, 1] *= scale2
        target[:, 2] *= scale3

        return target

    def load_depth(self, depth_path):
        depth = cv2.imread(depth_path, -1)

        if len(depth.shape) == 3:
            depth16 = np.uint16(depth[:, :, 1]*256) + np.uint16(depth[:, :, 2])
            depth16 = depth16.astype(np.uint16)
        elif len(depth.shape) == 2 and depth.dtype == 'uint16':
            depth16 = depth
        else:
            assert False, '[ Error ]: Unsupported depth type.'

        return depth16

    def get_pose(self, choose_frame, choose_obj):
        pose = {}
        input_file = open('{0}/{1}_pose.txt'.format(choose_obj, choose_frame), 'r')
        for i in range(4):
            input_line = input_file.readline()
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            input_line = input_line.split(' ')
            pose[choose_obj].append([float(input_line[0]), float(input_line[1]), float(input_line[2]), float(input_line[3])])
        ans = pose[choose_obj]
        ans = np.array(ans)
        ans_r = ans[:3, :3]
        ans_t = ans[:3, 3].flatten()
        input_file.close()
        ans_idx = choose_obj
        return ans_r, ans_t, ans_idx


    def get_frame(self, choose_frame, choose_obj, syn_or_real):

        img = Image.open('{0}_pose.png'.format(choose_frame))
        depth = np.array(self.load_depth('{0}_depth.png'.format(choose_frame)))
        # get the pose H from pose.txt
        target_r, target_t, idx = self.get_pose(choose_frame, choose_obj)

        cam_cx = self.cam_cx_1
        cam_cy = self.cam_cy_1
        cam_fx = self.cam_fx_1
        cam_fy = self.cam_fy_1
        cam_scale = 1.0

        # current idea:
        # for each item in tool category, we have multiple frames in the following structure,
        # right now, we just have frame fr and to
        # generate depth and mask for fr and to
        # this gives us a pair of info which can then be feed into model and get score feedback to train kp generator
        # although kpt generator generates kpts aimming to track obj through multiview loss
        # it will still get updated by Q loss
        # TODO: figure out what is model scales, is it bbox?
        #                      what is target * target pose? pose in next frame? looks more like in current frame
        #                      need to create model scales for each image pose (if put the tool in a way that its
        #                      always in camera, then still need to find a way to have model scale)
        #                      need to relate model scale with our item image in "multiple frames"
        #                      need to have depth and mask for each item in each frame, and properly linked
        #                      figure out what is choose and what is cloud
        #                      need to have a mesh file
        target = []
        input_file = open('{0}/model_scales/{1}.txt'.format(self.root, choose_obj), 'r')
        for i in range(8):
            input_line = input_file.readline()
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            input_line = input_line.split(' ')
            target.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        input_file.close()
        target = np.array(target)
        # put the target into a unit cube box and enlarge it to 1.3
        target = self.enlarge_bbox(copy.deepcopy(target))

        # make small disturbance
        delta = math.pi / 10.0
        noise_trans = 0.05
        # sample a small disturbance for point cloud
        r = euler_matrix(random.uniform(-delta, delta), random.uniform(-delta, delta), random.uniform(-delta, delta))[:3, :3]
        # why noise * 1000?
        t = np.array([random.uniform(-noise_trans, noise_trans) for i in range(3)]) * 1000.0

        # disturb the target (3d bbox) by noise
        target_tmp = target - (np.array([random.uniform(-noise_trans, noise_trans) for i in range(3)]) * 3000.0)
        # apply the transform acquired from: target_r, target_t, idx = self.get_pose(choose_frame, choose_obj)
        target_tmp = np.dot(target_tmp, target_r.T) + target_t
        # make first line and second line negative to place it at origin?
        target_tmp[:, 0] *= -1.0
        target_tmp[:, 1] *= -1.0
        # get 2d bbox in pixel coordinates from transformed target(3d bbox)
        rmin, rmax, cmin, cmax = get_2dbbox(target_tmp, cam_cx, cam_cy, cam_fx, cam_fy, cam_scale)
        # find the max length
        limit = search_fit(target)

        if self.add_noise:
            img = self.trancolor(img)

            if random.randint(1, 20) > 3:
                back_frame = random.sample(self.back_list, 1)[0]

                back_img = np.array(self.trancolor(Image.open(back_frame).resize((640, 480), Image.ANTIALIAS)))
                back_img = np.transpose(back_img, (2, 0, 1))
                # choose_frame is path to image of same item of multiple frames
                # mask = (cv2.imread('{0}_mask.png'.format(choose_frame))[:, :, 0] == 255)
                mask = (cv2.imread('{0}/data/{1}_mask.png'.format(self.root, choose_frame))[:, :, 0] == 255)
                img = np.transpose(np.array(img), (2, 0, 1))

                img = img * (~mask) + back_img * mask

                img = np.transpose(img, (1, 2, 0))

                back_cate_id = random.sample([1, 2, 3, 4, 5, 6], 1)[0]
                back_depth_choose_obj = random.sample(self.real_obj_name_list[back_cate_id], 1)[0]
                back_choose_frame = random.sample(self.real_obj_list[back_cate_id][back_depth_choose_obj], 1)[0]
                # back_depth = np.array(self.load_depth('{0}_depth.png'.format(back_choose_frame)))
                back_depth = np.array(self.load_depth('{0}/data/{1}_depth.png'.format(self.root, back_choose_frame)))

                ori_back_depth = back_depth * mask
                ori_depth = depth * (~mask)

                back_delta = ori_depth.flatten()[ori_depth.flatten() != 0].mean() - ori_back_depth.flatten()[ori_back_depth.flatten() != 0].mean()
                back_depth = back_depth + back_delta

                depth = depth * (~mask) + back_depth * mask

            else:
                img = np.array(img)
        else:
            img = np.array(img)

        # mask_target = (cv2.imread('{0}_mask.png'.format(choose_frame))[:, :, 2] == idx)[rmin:rmax, cmin:cmax]
        mask_target = (cv2.imread('{0}/data/{1}_mask.png'.format(self.root, choose_frame))[:, :, 2] == idx)[rmin:rmax, cmin:cmax]

        choose = (mask_target.flatten() != False).nonzero()[0]
        if len(choose) == 0:
            return 0

        img = np.transpose(img[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]
        depth = depth[rmin:rmax, cmin:cmax]

        img = img / 255.0

        choose = (depth.flatten() > -1000.0).nonzero()[0]
        depth_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((-pt0, -pt1, pt2), axis=1)

        cloud = np.dot(cloud - target_t, target_r)
        cloud = np.dot(cloud, r.T) + t

        choose_temp = (cloud[:, 0] > limit[0]) * (cloud[:, 0] < limit[1]) * (cloud[:, 1] > limit[2]) * (cloud[:, 1] < limit[3]) * (cloud[:, 2] > limit[4]) * (cloud[:, 2] < limit[5])

        choose = ((depth.flatten() != 0.0) * choose_temp).nonzero()[0]
        if len(choose) == 0:
            return 0
        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

        depth_masked = depth.flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        pt2 = depth_masked / cam_scale
        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy
        cloud = np.concatenate((-pt0, -pt1, pt2), axis=1)
        choose = np.array([choose])

        cloud = np.dot(cloud - target_t, target_r)
        cloud = np.dot(cloud, r.T) + t

        t = t / 1000.0
        cloud = cloud / 1000.0
        target = target / 1000.0

        return img, choose, cloud, r, t, target, mask_target


    def re_scale(self, target_fr, target_to):
        ans_scale = target_fr / target_to
        ans_target = target_fr
        ans_scale = ans_scale[0][0]

        return ans_target, ans_scale


    def __getitem__(self, index):
        # in our case, we dont have real obj now
        # 75% true, use real, 25% false, use syn
        # syn_or_real = (random.randint(1, 20) < 15)
        # if val, false, use real obj
        if self.mode == 'val':
            syn_or_real = False

        if self.mode == 'train':
            syn_or_real = True

        if syn_or_real:
            while 1:
                try:
                    choose_obj = random.sample(self.obj_name_list[self.cate_id], 1)[0]
                    # can still use 2 if there are only 2
                    choose_frame = random.sample(self.obj_list[self.cate_id][choose_obj], 2)
                    img_fr, choose_fr, cloud_fr, r_fr, t_fr, target_fr, mesh_pts_fr, mesh_bbox_fr, mask_target = self.get_frame(
                        choose_frame[0], choose_obj, syn_or_real)
                    if np.max(abs(target_fr)) > 1.0:
                        continue
                    img_to, choose_to, cloud_to, r_to, t_to, target_to, _, _, _, = self.get_frame(choose_frame[1],
                                                                                                  choose_obj,
                                                                                                  syn_or_real)
                    if np.max(abs(target_to)) > 1.0:
                        continue
                    target, scale_factor = self.re_scale(target_fr, target_to)
                    target_mesh_fr, scale_factor_mesh_fr = self.re_scale(target_fr, mesh_bbox_fr)
                    cloud_to = cloud_to * scale_factor
                    # why this mech is not used?
                    mesh = mesh_pts_fr * scale_factor_mesh_fr
                    t_to = t_to * scale_factor
                    break
                except:
                    continue

        else:
            while 1:
                try:
                    # randomly choose a frame, might not be consecutive, could be a huge time gap
                    # change to our fr and to
                    # nothing to be sample from
                    # choose_obj is an item from category cate_id
                    choose_obj = random.sample(self.real_obj_name_list[self.cate_id], 1)[0]
                    ###choose_frame = random.sample(self.real_obj_list[self.cate_id][choose_obj], 2)
                    # could be more in the future, need path
                    choose_frame = self.real_obj_list[self.cate_id][choose_obj]
                    choose_frame = [, "to"]
                    # img_fr: the start image
                    # choose_fr: mask if nonzero
                    # cloud_fr: point clound
                    # r_fr: small rotational disturbance
                    # t_fr: small translational disturbance
                    # target: 3d bbox
                    img_fr, choose_fr, cloud_fr, r_fr, t_fr, target, _ = self.get_frame(choose_frame[0], choose_obj, syn_or_real)
                    img_to, choose_to, cloud_to, r_to, t_to, target, _ = self.get_frame(choose_frame[1], choose_obj, syn_or_real)
                    if np.max(abs(target)) > 1.0:
                        continue
                    break
                except:
                    continue
        # save information
        if False:
            p_img = np.transpose(img_fr, (1, 2, 0))
            scipy.misc.imsave('temp/{0}_img_fr.png'.format(index), p_img)

            p_img = np.transpose(img_to, (1, 2, 0))
            scipy.misc.imsave('temp/{0}_img_to.png'.format(index), p_img)

            scipy.misc.imsave('temp/{0}_mask_fr.png'.format(index), mask_target.astype(np.int64))

            fw = open('temp/{0}_cld_fr.xyz'.format(index), 'w')
            for it in cloud_fr:
               fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            fw.close()

            fw = open('temp/{0}_cld_to.xyz'.format(index), 'w')
            for it in cloud_to:
               fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            fw.close()


        class_gt = np.array([self.cate_id-1])

        anchor_box, scale = self.get_anchor_box(target)
        cloud_fr, cloud_to = self.change_to_scale(scale, cloud_fr, cloud_to)

        mesh = self.mesh * scale

        if False:
            fw = open('temp/{0}_aft_cld_fr.xyz'.format(index), 'w')
            for it in cloud_fr:
               fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            fw.close()

            fw = open('temp/{0}_aft_cld_to.xyz'.format(index), 'w')
            for it in cloud_to:
               fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            fw.close()

            fw = open('temp/{0}_cld_mesh.xyz'.format(index), 'w')
            for it in mesh:
               fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            fw.close()

            fw = open('temp/{0}_target.xyz'.format(index), 'w')
            for it in target:
               fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            fw.close()

            fw = open('temp/{0}_anchor.xyz'.format(index), 'w')
            for it in anchor_box:
               fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            fw.close()

            fw = open('temp/{0}_small_anchor.xyz'.format(index), 'w')
            for it in small_anchor_box:
               fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            fw.close()

            fw = open('temp/{0}_pose_fr.xyz'.format(index), 'w')
            for it in r_fr:
               fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            it = t_fr
            fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            fw.write('{0}\n'.format(choose_frame[0]))
            fw.close()

            fw = open('temp/{0}_pose_to.xyz'.format(index), 'w')
            for it in r_to:
               fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            it = t_to
            fw.write('{0} {1} {2}\n'.format(it[0], it[1], it[2]))
            fw.write('{0}\n'.format(choose_frame[1]))
            fw.close()


        return self.norm(torch.from_numpy(img_fr.astype(np.float32))), \
               torch.LongTensor(choose_fr.astype(np.int32)), \
               torch.from_numpy(cloud_fr.astype(np.float32)), \
               torch.from_numpy(r_fr.astype(np.float32)), \
               torch.from_numpy(t_fr.astype(np.float32)), \
               self.norm(torch.from_numpy(img_to.astype(np.float32))), \
               torch.LongTensor(choose_to.astype(np.int32)), \
               torch.from_numpy(cloud_to.astype(np.float32)), \
               torch.from_numpy(r_to.astype(np.float32)), \
               torch.from_numpy(t_to.astype(np.float32)), \
               torch.from_numpy(mesh.astype(np.float32)), \
               torch.from_numpy(anchor_box.astype(np.float32)), \
               torch.from_numpy(scale.astype(np.float32)), \
               torch.LongTensor(class_gt.astype(np.int32))

    def __len__(self):
        return self.length


border_list = [-1, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

def get_2dbbox(cloud, cam_cx, cam_cy, cam_fx, cam_fy, cam_scale):
    rmin = 10000
    rmax = -10000
    cmin = 10000
    cmax = -10000
    for tg in cloud:
        p1 = int(tg[0] * cam_fx / tg[2] + cam_cx)
        p0 = int(tg[1] * cam_fy / tg[2] + cam_cy)
        if p0 < rmin:
            rmin = p0
        if p0 > rmax:
            rmax = p0
        if p1 < cmin:
            cmin = p1
        if p1 > cmax:
            cmax = p1
    rmax += 1
    cmax += 1
    if rmin < 0:
        rmin = 0
    if cmin < 0:
        cmin = 0
    if rmax >= 480:
        rmax = 479
    if cmax >= 640:
        cmax = 639

    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
        
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt

    if ((rmax-rmin) in border_list) and ((cmax-cmin) in border_list):
        return rmin, rmax, cmin, cmax
    else:
        return 0


def search_fit(points):
    min_x = min(points[:, 0])
    max_x = max(points[:, 0])
    min_y = min(points[:, 1])
    max_y = max(points[:, 1])
    min_z = min(points[:, 2])
    max_z = max(points[:, 2])

    return [min_x, max_x, min_y, max_y, min_z, max_z]
