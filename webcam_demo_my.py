import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from opt import opt

from dataloader_webcam_my import WebcamLoader, DetectionLoader, DetectionProcessor, DataWriter, crop_from_dets, Mscoco
from yolo.darknet import Darknet
from yolo.util import write_results, dynamic_write_results
from SPPE.src.main_fast_inference import *

from SPPE.src.utils.img import im_to_torch
import os
import sys
from tqdm import tqdm
import time
from fn import getTime
import cv2
import clientdemo.Conf as Conf
from clientdemo.DataModel import *
import clientdemo.HttpHelper as HttpHelper
import time
from pPose_nms import write_json

from align import AlignPoints

args = opt
args.dataset = 'coco'


def loop():
    n = 0
    while True:
        yield n
        n += 1


class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 4)
        self.drop = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def save_db(pose_info, classidx):
    classidx = classidx.item()
    print(classidx)
    print(pose_info.status)
    if pose_info.status is None:
        if classidx == 0:
            pose_info.timesit = 1 / 24
            pose_info.isalarm = False
        elif classidx == 1:
            pose_info.timelie = 1 / 24
            pose_info.isalarm = True
        elif classidx == 2:
            pose_info.timestand = 1 / 24
            pose_info.isalarm = False
        elif classidx == 3:
            pose_info.timedown = 1 / 24
            pose_info.isalarm = True
        pose_info.date = time.strftime('%Y-%m-%dT00:00:00', time.localtime())
        pose_info.status = classidx
        return pose_info

    if pose_info.status == classidx:
        if classidx == 0:
            pose_info.timesit += 1 / 24
            pose_info.isalarm = False
        elif classidx == 1:
            pose_info.timelie += 1 / 24
            pose_info.isalarm = True
        elif classidx == 2:
            pose_info.timestand += 1 / 24
            pose_info.isalarm = False
        elif classidx == 3:
            pose_info.timedown += 1 / 24
            pose_info.isalarm = True
        pose_info.date = time.strftime('%Y-%m-%dT00:00:00', time.localtime())
    else:
        if classidx == 0:
            pose_info.timesit = 1 / 24
            pose_info.isalarm = False
        elif classidx == 1:
            pose_info.timelie = 1 / 24
            pose_info.isalarm = True
        elif classidx == 2:
            pose_info.timestand = 1 / 24
            pose_info.isalarm = False
        elif classidx == 3:
            pose_info.timedown = 1 / 24
            pose_info.isalarm = True
        pose_info.date = time.strftime('%Y-%m-%dT00:00:00', time.localtime())
        pose_info.status = classidx

    return pose_info


class ParsePoseDemo:
    def __init__(self, camera, out_video_path, detbatch, pose_model, pos_reg_model, save_video=False):
        self.camera_info = camera
        self.output_path = out_video_path
        self.detbatch = detbatch
        self.pose_model = pose_model
        self.pose_reg_model = pos_reg_model
        self.save_video = save_video

    def parse(self):
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        data_loader = WebcamLoader(self.camera_info.videoAddress).start()
        (fourcc, fps, frameSize) = data_loader.videoinfo()

        print('Loading YOLO model..')
        sys.stdout.flush()
        det_loader = DetectionLoader(data_loader, batchSize=self.detbatch).start()
        det_processor = DetectionProcessor(det_loader).start()

        aligner = AlignPoints()

        # Data writer
        # save_path = os.path.join(args.outputpath, 'AlphaPose_webcam' + webcam + '.avi')
        if self.save_video:
            writer = DataWriter(self.save_video, self.output_path, cv2.VideoWriter_fourcc(*'XVID'),
                            fps, frameSize, pos_reg_model=pos_reg_model, aligner=aligner).start()

        # 不明白是何用途，请添加注释
        runtime_profile = {
            'dt': [],
            'pt': [],
            'pn': []
        }

        sys.stdout.flush()
        batch_size = self.detbatch
        for i in tqdm(loop()):  # 进度显示
            try:
                start_time = getTime()
                with torch.no_grad():
                    (inps, orig_img, im_name, boxes, scores, pt1, pt2) = det_processor.read()
                    if boxes is None or boxes.nelement() == 0:
                        if self.save_video:
                            writer.save(None, None, None, None, None, orig_img, im_name.split('/')[-1])
                        continue

                    ckpt_time, det_time = getTime(start_time)
                    runtime_profile['dt'].append(det_time)
                    # Pose Estimation

                    datalen = inps.size(0)
                    leftover = 0
                    if (datalen) % batch_size:
                        leftover = 1
                    num_batches = datalen // batch_size + leftover
                    hm = []
                    for j in range(num_batches):
                        inps_j = inps[j * batch_size:min((j + 1) * batch_size, datalen)].cuda()
                        hm_j = pose_model(inps_j)
                        hm.append(hm_j)
                    hm = torch.cat(hm)
                    ckpt_time, pose_time = getTime(ckpt_time)

                    hm = hm.cpu().data
                    if self.save_video:
                        writer.save(boxes, scores, hm, pt1, pt2, orig_img, im_name.split('/')[-1])
                    while not writer.result_Q.empty():
                        boxes, classidx = writer.result_Q.get()
                        print('classidx:', classidx)
                        pose_info = save_db(pose_info, classidx)
                        pose_url = Conf.Urls.PoseInfoUrl + '/UpdateOrCreatePoseInfo'
                        print(pose_url)
                        HttpHelper.create_item(pose_url, pose_info)
                        print('ok')

                    ckpt_time, post_time = getTime(ckpt_time)
            except KeyboardInterrupt:
                break

        while (writer.running()):
            pass
        writer.stop()


if __name__ == "__main__":
    # Load pose model
    pose_dataset = Mscoco()
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    pose_model.cuda()
    pose_model.eval()
    pos_reg_model = NeuralNet(17 * 3 * 9).cuda()
    pos_reg_model.load_state_dict(torch.load('exps\\42_model.ckpt'))
    pos_reg_model.eval()
    pass
