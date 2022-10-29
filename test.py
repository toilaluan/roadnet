import argparse
import torch
import torch.nn as nn
from data.road_dataset import RoadDataset
from torch.utils.data import DataLoader
from model.u2net import U2NET, U2NETP
from model.build_resnet_decoder import ResnetSeg
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import cv2
import argparse
import os
import shutil
def test(opt):
    dataset = RoadDataset(opt.data_path)
    test_indices = SubsetRandomSampler(opt.indices)
    test_loader = DataLoader(dataset, 1, sampler = test_indices)
    # model = U2NET(3, 1)
    model = ResnetSeg()
    model.load_state_dict(torch.load(opt.pretrain))
    device = torch.device(opt.device)
    model.to(device)
    shutil.rmtree(opt.save_path)
    os.makedirs(opt.save_path)
    for i, data in enumerate(test_loader):
        images = data['image'].to(device)
        segments = data['segment'].to(device)
        outputs = model(images)[0]
        outputs = outputs > opt.threshold
        prediction = np.array(outputs[0].cpu().permute(1,2,0)*255).astype(np.uint8)
        label = np.array(segments[0].cpu().permute(1,2,0)*255).astype(np.uint8)
        pred_label_img = cv2.vconcat([prediction, label])
        cv2.imwrite("{}/{}.png".format(opt.save_path, i), pred_label_img)
        
    # print(model)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default= "Ottawa-Dataset")
    parser.add_argument("--indices", default = [0,15,16,17,18,19])
    # parser.add_argument("--pretrain", default= "/home/aimenext/luantt/road_seg/weights/u2net_300_epoch.pt")
    parser.add_argument("--pretrain", default= "/home/aimenext/luantt/road_seg/weights/resnet_fcloss_epoch_450.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_path", default="predictions")
    parser.add_argument("--threshold", default= 0.4)
    opt = parser.parse_args()
    test(opt)