import warnings
warnings.filterwarnings("ignore")
from data import *
from data.road_dataset import RoadDataset
from model import *
import torch
import torch.optim as optimizer
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import numpy as np
from model.u2net import U2NET, U2NETP
from model.build_resnet_decoder import ResnetSeg
from torch.utils.data.sampler import SubsetRandomSampler
import cv2 
from loss.balanced_ce_loss import BalanceCrossEntropyLoss
from loss.focal_loss import FocalLoss, BinaryFocalLoss
from utils.utils import count_parameters
from sklearn.metrics import precision_score, f1_score
from torch.utils.tensorboard import SummaryWriter
def _get_balanced_sigmoid_cross_entropy(self,x):
    count_neg = torch.sum(1. - x)
    count_pos = torch.sum(x)
    beta = count_neg / (count_neg + count_pos)
    pos_weight = beta / (1 - beta)
    cost = torch.nn.BCEWithLogitsLoss(size_average=True, reduce=True, pos_weight=pos_weight)
    return cost, 1-beta
writer = SummaryWriter("runs/resnet_112_focal")
model = ResnetSeg()
model.load_state_dict(torch.load('/home/aimenext/luantt/road_seg/weights/resnet_112_fcloss_epoch_200.pt'))
# model.load_state_dict(torch.load("/home/aimenext/luantt/road_seg/weights/last_26_10.pt"))
count_parameters(model)
road_dataset = RoadDataset('/home/aimenext/luantt/road_seg/Ottawa-Dataset')

indices = list(range(len(road_dataset)))

np.random.shuffle(indices)

train_indices = SubsetRandomSampler([1,2,3,4,5,6,7,8,9,10,11,12,13,14])
val_indices = SubsetRandomSampler([0,15,16,17,18,19])
# train_indices = SubsetRandomSampler(indices[:17])
# val_indices = SubsetRandomSampler(indices[17:])
train_loader = DataLoader(road_dataset, batch_size = 16, sampler = train_indices)
val_loader = DataLoader(road_dataset, batch_size=1, sampler=val_indices)
# criterion = FocalLoss(reduction='mean', gamma=2)
criterion = BinaryFocalLoss()
# criterion = torch.nn.BCEWithLogitsLoss()
optim = optimizer.Adam(params = model.parameters(), lr=1e-4)
scheduler = lr_scheduler.StepLR(optim, gamma = 0.1, step_size = 100)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
epochs = 500
def validate(threshold = 0.5):
    device = torch.device("cpu")
    model.eval()
    model.to(device)
    preds = []
    labels = []
    n_iter = 1
    for _ in range(n_iter):
        for i, data in enumerate(val_loader):
            images = data['image'].to(device)
            segments = data['segment'].to(device)
            # print(segments.unique())
            outputs = model(images)
            outputs = outputs > threshold
            preds.extend(segments.flatten().tolist())
            labels.extend(outputs.flatten().tolist())
            # print(outputs.shape)
            segment = np.array(segments[0].permute(1,2,0).cpu()*255).astype(np.uint8)
            out_segment = np.array((outputs[0] > 0.5).permute(1,2,0).cpu()*255).astype(np.uint8)
            # print(segment.shape, out_segment.shape)
            debug = cv2.hconcat([out_segment, segment])
            # cv2.imwrite("label.png", segment)
            cv2.imwrite("debugs/debug_{}.png".format(i), debug)
    return precision_score(preds, labels), f1_score(preds, labels)
bce_loss = torch.nn.BCELoss()

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):

	loss0 = bce_loss(d0,labels_v)
	loss1 = bce_loss(d1,labels_v)
	loss2 = bce_loss(d2,labels_v)
	loss3 = bce_loss(d3,labels_v)
	loss4 = bce_loss(d4,labels_v)
	loss5 = bce_loss(d5,labels_v)
	loss6 = bce_loss(d6,labels_v)

	loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
	print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n"%(loss0.data.item(),loss1.data.item(),loss2.data.item(),loss3.data.item(),loss4.data.item(),loss5.data.item(),loss6.data.item()))

	return loss0, loss
# def cal_loss_more_on_edge()
for epoch in range(epochs):
    
    print('-'*10 + 'Training Epoch {}'.format(epoch+1))
    model.train()
    model.to(device)
    for i, data in enumerate(train_loader):
        optim.zero_grad()
        images = data['image'].to(device)
        segments = data['segment'].to(device)
        # d0,d1,d2,d3,d4,d5,d6 = model(images)
        # loss0, loss = muti_bce_loss_fusion(d0,d1,d2,d3,d4,d5,d6, segments)
        outputs = model(images)
        # print(outputs.shape)
        # print(segments.shape)
        loss = criterion(outputs, segments)
        loss.backward()
        optim.step()
        print('Epoch {} - LR [{}] - Minibatch [{}/{}] - Loss [{}]'.format(epoch, scheduler.get_lr(), i, len(train_loader), loss.item()))
        # del d0, d1, d2, d3, d4, d5, d6, loss0, loss
    scheduler.step()
    if epoch % 50 == 0:
        prec_s, f1_s = validate()
        print('Epoch {} - Precision_score [{}] - F1_score [{}]'.format(epoch, prec_s, f1_s))
        writer.add_scalars("Metrics", {"precision_score" : prec_s, "f1_score" : f1_s}, epoch)
        torch.save(model.state_dict(), "weights/resnet_112_fcloss_epoch_{}.pt".format(epoch))
    writer.add_scalar("Loss", loss.item(), epoch)

