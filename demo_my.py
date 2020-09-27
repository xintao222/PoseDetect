import os
from time import time
import json

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np

from opt import opt
from SPPE.src.utils.img import load_image, cropBox, im_to_torch, torch_to_im
from SPPE.src.utils.eval import getPrediction
from pPose_nms import pose_nms
from yolo.darknet import Darknet
from yolo.util import dynamic_write_results
from dataloader import Mscoco
from SPPE.src.main_fast_inference import InferenNet, InferenNet_fast
from align import AlignPoints

args = opt
args.dataset = 'coco'


class ImageLoader:
    def __init__(self, format='yolo'):
        self.format = format
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def getitem_ssd(self, im_path):
        im = Image.open(im_path)
        if im.mode == 'L':
            im = im.convert('RGB')
        inp = load_image(im_path)
        ow = oh = 512
        im = im.resize((ow, oh))
        im = self.transform(im)

        return (im, inp, im_path)

    def getitem_yolo(self, im_path):
        inp_dim = int(opt.inp_dim)
        im, ori_im, im_dim_list = self.prep_image(im_path, inp_dim)

        ims = [im]
        ori_ims = [ori_im]
        im_dim_lists = [im_dim_list]
        im_names = [os.path.basename(im_path)]
        with torch.no_grad():
            # Human Detection
            ims = torch.cat(ims)
            im_dim_lists = torch.FloatTensor(im_dim_lists).repeat(1, 2)
        return (ims, ori_ims, im_names, im_dim_lists)

    def letterbox_image(self, img, inp_dim):
        '''resize image with unchanged asepect retio using padding'''
        img_w, img_h = img.shape[1], img.shape[0]
        w, h = inp_dim
        new_w = int(img_w * min(w / img_w, h / img_h))
        new_h = int(img_h * min(w / img_w, h / img_h))
        resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)
        canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h,
        (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image
        return canvas

    def prep_image(self, img, inp_dim):
        orig_im = cv2.imread(img)
        dim = orig_im.shape[1], orig_im.shape[0]
        img = (self.letterbox_image(orig_im, (inp_dim, inp_dim)))
        img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
        img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
        return img_, orig_im, dim


class DetectionLoader:
    def __init__(self, dataloder):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.det_model = Darknet("yolo/cfg/yolov3-spp.cfg")
        self.det_model.load_weights('models/yolo/yolov3-spp.weights')
        self.det_model.net_info['height'] = opt.inp_dim
        self.det_inp_dim = int(self.det_model.net_info['height'])
        assert self.det_inp_dim % 32 == 0
        assert self.det_inp_dim > 32
        self.det_model.cuda()
        self.det_model.eval()

        self.stopped = False
        self.dataloder = dataloder

    def detect_image(self, im_path):
        im, ori_im, im_name, im_dim_list = self.dataloder.getitem_yolo(im_path)

        with torch.no_grad():
            im = im.cuda()
        prediction = self.det_model(im, CUDA=True)
        # NMS process
        dets = dynamic_write_results(prediction, opt.confidence,
                                     opt.num_classes, nms=True,
                                     nms_conf=opt.nms_thesh)
        if isinstance(dets, int) or dets.shape[0] == 0:
            return (ori_im[0], im_name[0], None, None, None, None, None)

        dets = dets.cpu()
        im_dim_list = torch.index_select(im_dim_list, 0, dets[:, 0].long())
        scaling_factor = torch.min(self.det_inp_dim / im_dim_list, 1)[0] \
            .view(-1, 1)
        # coordinate transfer
        dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
        dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

        dets[:, 1: 5] /= scaling_factor
        for j in range(dets.shape[0]):
            dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0,
                                          im_dim_list[j, 0])
            dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0,
                                          im_dim_list[j, 1])
        boxes = dets[:, 1:5]
        scores = dets[:, 5:6]

        if boxes.shape[0] > 1:
            boxes = boxes[scores.argmax()].unsqueeze(0)
            scores = scores[scores.argmax()].unsqueeze(0)
            dets = dets[scores.argmax()].unsqueeze(0)
        # len(ori_im) === 1
        for k in range(len(ori_im)):

            boxes_k = boxes[dets[:, 0] == k]
            if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
                return (ori_im[k], im_name[k], None, None, None, None, None)
            inps = torch.zeros(boxes_k.size(0), 3,
                               opt.inputResH, opt.inputResW)
            pt1 = torch.zeros(boxes_k.size(0), 2)
            pt2 = torch.zeros(boxes_k.size(0), 2)
            return (ori_im[k], im_name[k], boxes_k, scores[dets[:, 0] == k],
                    inps, pt1, pt2)


class DetectionProcessor:

    def __init__(self, detectionLoader):
        self.detection_loader = detectionLoader

    def detect_image(self, im_path, outputdir):
        with torch.no_grad():
            ori_im, im_name, boxes, scores, inps, pt1, pt2 = \
                self.detection_loader.detect_image(im_path)

            if ori_im is None:
                return (None, None, None, None, None, None, None)
            if boxes is None or boxes.nelement() == 0:
                return (None, ori_im, im_name, boxes, scores, None, None)
            inp = im_to_torch(cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB))
            inps, pt1, pt2 = self.crop_from_dets(inp, boxes, inps, pt1, pt2)
            return (inps, ori_im, im_name, boxes, scores, pt1, pt2)

    def crop_from_dets(self, img, boxes, inps, pt1, pt2):

        imght = img.size(1)
        imgwidth = img.size(2)
        tmp_img = img

        tmp_img[0].add_(-0.406)
        tmp_img[1].add_(-0.457)
        tmp_img[2].add_(-0.480)
        for i, box in enumerate(boxes):
            upLeft = torch.Tensor((float(box[0]), float(box[1])))
            bottomRight = torch.Tensor((float(box[2]), float(box[3])))
            ht = bottomRight[1] - upLeft[1]
            width = bottomRight[0] - upLeft[0]
            if width > 100:
                scaleRate = 0.2
            else:
                scaleRate = 0.3
            upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
            upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
            bottomRight[0] = max(
                min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2),
                upLeft[0] + 5
            )
            bottomRight[1] = max(
                min(imght - 1, bottomRight[1] + ht * scaleRate / 2),
                upLeft[1] + 5
            )
            try:
                inps[i] = cropBox(tmp_img, upLeft, bottomRight,
                                  opt.inputResH, opt.inputResW)
            except IndexError:
                print(tmp_img.shape)
                print(upLeft)
                print(bottomRight)
                print('===')

            pt1[i] = upLeft
            pt2[i] = bottomRight
            return inps, pt1, pt2


def fetch_result(boxes, scores, hm_data, pt1, pt2, ori_im, im_name):
    ori_im = np.array(ori_im, dtype=np.uint8)
    if boxes is None:
        return None
    preds_hm, preds_img, preds_scores = getPrediction(hm_data, pt1, pt2,
                                                      opt.inputResH,
                                                      opt.inputResW,
                                                      opt.outputResH,
                                                      opt.outputResW)
    result = pose_nms(boxes, scores, preds_img, preds_scores)
    result = {
        'imgname': im_name,
        'result': result,
        'bbox': boxes,
    }
    return result


def generate_json(all_results, for_eval=False):
    form = opt.format
    json_results = []
    json_results_cmu = {}
    for im_res in all_results:
        im_name = im_res['imgname']
        # only support single person per image!!
        bbox = im_res['bbox'].numpy().tolist()
        result = {}
        result['bbox'] = bbox
        result['class'] = im_res['class'].numpy().tolist()

        for human in im_res['result']:
            keypoints = []
            if for_eval:
                result['image_id'] = \
                    int(im_name.split('/')[-1].split('.')[0].split('_')[-1])
            else:
                result['image_id'] = im_name.split('/')[-1]
            result['category_id'] = 1
            result['bbox'] = bbox
            kp_preds = human['keypoints']
            kp_scores = human['kp_score']
            pro_scores = human['proposal_score']
            for n in range(kp_scores.shape[0]):
                keypoints.append(float(kp_preds[n, 0]))
                keypoints.append(float(kp_preds[n, 1]))
                keypoints.append(float(kp_scores[n]))
            result['keypoints'] = keypoints
            result['score'] = float(pro_scores)
            if form == 'cmu':  # the form of CMU - Pose
                if result['image_id'] not in json_results_cmu.keys():
                    json_results_cmu[result['image_id']] = {}
                    json_results_cmu[result['image_id']]['version'] = "AlphaPose v0.2"
                    json_results_cmu[result['image_id']]['bodies'] = []
                tmp = {'joints': []}
                result['keypoints'].append(
                    (result['keypoints'][15] + result['keypoints'][18]) / 2)
                result['keypoints'].append(
                    (result['keypoints'][16] + result['keypoints'][19]) / 2)
                result['keypoints'].append(
                    (result['keypoints'][17] + result['keypoints'][20]) / 2)
                indexarr = [0, 51, 18, 24, 30, 15, 21, 27,
                            36, 42, 48, 33, 39, 45, 6, 3, 12, 9]
                for i in indexarr:
                    tmp['joints'].append(result['keypoints'][i])
                    tmp['joints'].append(result['keypoints'][i + 1])
                    tmp['joints'].append(result['keypoints'][i + 2])
                json_results_cmu[result['image_id']]['bodies'].append(tmp)
            elif form == 'open':  # the form of OpenPose
                if result['image_id'] not in json_results_cmu.keys():
                    json_results_cmu[result['image_id']] = {}
                    json_results_cmu[result['image_id']]['version'] = "AlphaPose v0.2"
                    json_results_cmu[result['image_id']]['people'] = []
                tmp = {'pose_keypoints 2d': []}
                result['keypoints'].append(
                    (result['keypoints'][15] + result['keypoints'][18]) / 2)
                result['keypoints'].append(
                    (result['keypoints'][16] + result['keypoints'][19]) / 2)
                result['keypoints'].append(
                    (result['keypoints'][17] + result['keypoints'][20]) / 2)
                indexarr = [0, 51, 18, 24, 30, 15, 21, 27,
                            36, 42, 48, 33, 39, 45, 6, 3, 12, 9]
                for i in indexarr:
                    tmp['pose_keypoints_2d'].append(result['keypoints'][i])
                    tmp['pose_keypoints 2d'].append(result['keypoints'][i + 1])
                    tmp['pose_keypoints 2d'].append(result['keypoints'][i + 2])
                json_results_cmu[result['image_id']]['people'].append(tmp)
            else:
                json_results.append(result)
    return json_results


def run_folder(inputdir, output_json_path, im_names, detection_processor, pose_model,
                pos_reg_model,aligner, missed="./missed"):
    idx = 0
    im_len = len(im_names)
    result_list = []
    for im_name in im_names:
        im_path = os.path.join(inputdir, im_name)
        temp_kps = []
        with torch.no_grad():
            (inps, ori_im, im_name, boxes, scores, pt1, pt2) = \
                detection_processor.detect_image(im_path, output_json_path)
            # Modified
            if boxes is None or boxes.nelement() == 0:
                with open(missed, "a") as f:
                    f.write(im_path + "\n")
                # pass
            try:
                inps_j = inps[0: inps.size(0)].cuda()
            except AttributeError:
                continue
            hm_j = pose_model(inps_j)
            hm = torch.cat([hm_j])
            hm = hm.cpu()
            result = fetch_result(boxes, scores, hm, pt1, pt2, ori_im,
                                  os.path.basename(im_name))

            pos = result['result'][0]['keypoints'].unsqueeze(0).numpy()
            pos = aligner.align_points(pos)[0]
            pos = (pos[..., :2] - 129) / 255
            pos = torch.FloatTensor(pos)
            kp = torch.cat((pos, result['result'][0]['kp_score']), 1)
            kp = kp.unsqueeze(0)
            if len(temp_kps) < 9:
                kp = kp.reshape([1, -1]).cuda()
                temp_kps.append(kp)
                kp = kp.repeat(9, 1).reshape(1,-1)
                outputs = pos_reg_model(kp)
                _, preds = torch.max(outputs, 1)
                result['class'] = preds.cpu()
                result_list.append(result)
            else:
                temp_kps.append(kp)
                temp_kps.pop(0)
                _temp_kps = torch.cat(temp_kps)
                _temp_kps.cuda()
                _temp_kps = _temp_kps.reshape([1, -1])
                outputs = pos_reg_model(kp)
                preds = torch.max(outputs, 1)
                result['class'] = preds.cpu()
                result_list.append(result)
        idx += 1
        print('{}- {} / {}'.format(inputdir, idx, im_len))
    datas = generate_json(result_list)
    with open(output_json_path, 'w') as f:
        f.write(json.dumps(datas))


if __name__ == '__main__':
    # root_dir = 'D:\pycharm_workspace\PoseRecognition\exps\\test1\\tt\\'
    root_dir = 'exps\\test_im'
    dest_path = 'exps\\'
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    # Load pose model
    pose_dataset = Mscoco()
    if args.fast_inference:
        pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    else:
        pose_model = InferenNet(4 * 1 + 1, pose_dataset)
    pose_model.cuda()
    pose_model.eval()

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

    pos_reg_model = NeuralNet(17 * 3 * 9).cuda()
    pos_reg_model.load_state_dict(torch.load('exps\\42_model.ckpt'))
    pos_reg_model.eval()

    aligner = AlignPoints()
    runtime_profile = {
        'dt': [],
        'pt': [],
        'pn': []
    }

    image_loader = ImageLoader()
    detection_loader = DetectionLoader(image_loader)
    detection_processor = DetectionProcessor(detection_loader)
    is_file = os.path.isfile(os.path.join(root_dir, os.listdir(root_dir)[0]))
    if is_file:
        dirname = root_dir.split('\\')[-1]
        # outputdir ='./output'
        ext_filters = ('.jpg', '.png')
        im_names = filter(lambda f: os.path.splitext(
            f)[-1].lower() in ext_filters, os.listdir(root_dir))
        im_names = sorted(list(im_names))
        run_folder(root_dir, os.path.join(dest_path, dirname + '_alphapose_old.json'), im_names,
                   detection_processor, pose_model, pos_reg_model, aligner)
    else:
        for dirname in os.listdir(root_dir):
            inputdir = os.path.join(root_dir, dirname)
        # outputdir = './output'
        ext_filters = ('.jpg', '.png')
        im_names = filter(lambda f: os.path.splitext(f)[-1].lower() in ext_filters,
                          os.listdir(inputdir))
        im_names = sorted(list(im_names))
        run_folder(inputdir, os.path.join(dest_path, dirname + '_alphapose_old.json'), im_names,
                   detection_processor, pose_model)
