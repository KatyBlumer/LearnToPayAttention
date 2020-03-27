import os
import sys
sys.path.append(f"{os.getcwd()}/venv_t/lib/python3.7/site-packages")

import argparse
import collections
import cv2
# import matplotlib.pyplot as plt
import numpy as np
import random
# import tensorflow as tf
# from tensorflow.python.summary.summary_iterator import summary_iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.utils as utils
import torchvision.transforms as transforms

import sympy
from PIL import Image

parser = argparse.ArgumentParser(description="LearnToPayAttn-CIFAR100")

parser.add_argument("--BATCH_SIZE", type=int, default=128, help="batch size")
parser.add_argument("--EPOCHS", type=int, default=300, help="number of epochs")
parser.add_argument("--LR", type=float, default=0.1, help="initial learning rate")
parser.add_argument("--LOG_DIR", type=str, default="logs", help='path of log files')

parser.add_argument("--NORMALIZE_ATTN", action='store_true', help='if True, attention map is normalized by softmax; otherwise use sigmoid')
parser.add_argument("--USE_ATTN", action='store_false', help='turn down attention')
parser.add_argument("--LOG_IMAGES", action='store_false', help='log images and (is available) attention maps')

parser.add_argument("--EXAMPLE_TYPE", type=str, default="", help='Which type of task to train on')

OPT = parser.parse_args()

BASE_SEED = 0
torch.backends.cudnn.deterministic = True
torch.manual_seed(BASE_SEED)
torch.cuda.manual_seed_all(BASE_SEED)

# LOG_DIR = "logs/TEMP_CHANGE_THIS"

DATASET_SIZE = 1000
# BATCH_SIZE = 128
# EPOCHS = 300
# LR = 0.1

# NORMALIZE_ATTN = False
# USE_ATTN = True
# LOG_IMAGES = True  # False

NUM_AUG = 3
IM_SIZE = 32

#@title Attn visualizers
def visualize_attn_softmax(I, c, up_factor, nrow):
    # image
    img = I.permute((1,2,0)).cpu().numpy()
    # compute the heatmap
    N,C,W,H = c.size()
    a = F.softmax(c.view(N,C,-1), dim=2).view(N,C,W,H)
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=nrow, normalize=True, scale_each=True, pad_value=1)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    # add the heatmap to the image
    vis = 0.6 * img + 0.4 * attn
    return torch.from_numpy(vis).permute(2,0,1)

def visualize_attn_sigmoid(I, c, up_factor, nrow):
    # image
    img = I.permute((1,2,0)).cpu().numpy()
    # compute the heatmap
    a = torch.sigmoid(c)
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=nrow, normalize=False, pad_value=1)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    # add the heatmap to the image
    vis = 0.6 * img + 0.4 * attn
    return torch.from_numpy(vis).permute(2,0,1)

#@title Dataset drawing
def create_rgb(r, g, b):
  return np.dstack(
      (r * 255,
       g * 255,
       b * 255
       )
  ).astype(np.uint8)

def draw_circle(center, rad=7, thickness=4, imsize=32):
  xx, yy = np.mgrid[:imsize, :imsize]
  center_x, center_y = center
  circ = (xx - center_x) ** 2 + (yy - center_y) ** 2
  donut = np.logical_and(circ < (rad + thickness), circ > (rad - thickness))
  return donut

def random_circle(imsize):
  center_x = random.randrange(3, imsize - 3)
  center_y = random.randrange(3, imsize - 3)
  return draw_circle([center_x, center_y])


def create_presence_example(label, imsize):
  if label:
    circ = random_circle(imsize)
  else:
    circ = np.zeros([imsize, imsize])

  return create_rgb(circ, circ, circ)

def create_color_example(label, imsize):
  circ = random_circle(imsize)
  zeros = np.zeros([imsize, imsize])

  if label:
    return create_rgb(circ, zeros, zeros)
  else:
    return create_rgb(zeros, circ, zeros)

def create_location_example(label, imsize):
  center_x = random.randrange(3, (imsize / 2) - 3)
  center_y = random.randrange(3, imsize - 3)

  if label:
    center_x += (imsize / 2)

  circ = draw_circle([center_x, center_y])
  return create_rgb(circ, circ, circ)

def create_distance_example(label, imsize):
  dists = [5, 7]
  dist = dists[label]

  center_x = random.randrange(3, imsize - (3+max(dists)))
  center_y = random.randrange(3, imsize - (3+max(dists)))
  rad  = 7
  thickness = 4

  im_arr = create_rgb(
      # red
      np.zeros([32, 32]),
      # green
      draw_circle([center_x, center_y]),
      # blue
      draw_circle([center_x+dist, center_y+dist])
  )

  return Image.fromarray(im_arr)

EXAMPLE_TYPES = {
  'presence': create_presence_example,
  'color': create_color_example,
  'location': create_location_example,
  'distance': create_distance_example,
}


class DrawDataset(torch.utils.data.Dataset):

    def __init__(self, draw_func, transform=None, imsize=32):
        self.imsize=imsize
        self.transform = transform
        self.draw_func = draw_func

    def __len__(self):
        return 200

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print("Image idx", idx)

        label = random.randrange(2)

        im = self.draw_func(label, self.imsize)

        if self.transform:
            im = self.transform(im)

        return (im, label)


#@title Network pieces
def weights_init_xavierUniform(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight, a=0, b=1)
            nn.init.constant_(m.bias, val=0.)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, val=0.)

class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, num_conv, pool=False):
        super(ConvBlock, self).__init__()
        features = [in_features] + [out_features for i in range(num_conv)]
        layers = []
        for i in range(len(features)-1):
            layers.append(nn.Conv2d(in_channels=features[i], out_channels=features[i+1], kernel_size=3, padding=1, bias=True))
            layers.append(nn.BatchNorm2d(num_features=features[i+1], affine=True, track_running_stats=True))
            layers.append(nn.ReLU())
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.op = nn.Sequential(*layers)
    def forward(self, x):
        return self.op(x)

class ProjectorBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ProjectorBlock, self).__init__()
        self.op = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, padding=0, bias=False)
    def forward(self, inputs):
        return self.op(inputs)

class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)
    def forward(self, l, g):
        N, C, W, H = l.size()
        c = self.op(l+g) # batch_sizex1xWxH
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,W,H), g


#@title Network
class AttnVGG_after(nn.Module):
    def __init__(self, im_size, num_classes, attention=True, normalize_attn=True, init='xavierUniform'):
        super(AttnVGG_after, self).__init__()
        self.attention = attention
        # conv blocks
        self.conv_block1 = ConvBlock(3, 64, 2)
        self.conv_block2 = ConvBlock(64, 128, 2)
        self.conv_block3 = ConvBlock(128, 256, 3)
        self.conv_block4 = ConvBlock(256, 512, 3)
        self.conv_block5 = ConvBlock(512, 512, 3)
        self.conv_block6 = ConvBlock(512, 512, 2, pool=True)
        self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=int(im_size/32), padding=0, bias=True)
        # Projectors & Compatibility functions
        if self.attention:
            self.projector = ProjectorBlock(256, 512)
            self.attn1 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn2 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
            self.attn3 = LinearAttentionBlock(in_features=512, normalize_attn=normalize_attn)
        # final classification layer
        if self.attention:
            self.classify = nn.Linear(in_features=512*3, out_features=num_classes, bias=True)
        else:
            self.classify = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        # initialize
        if init == 'kaimingNormal':
            weights_init_kaimingNormal(self)
        elif init == 'kaimingUniform':
            weights_init_kaimingUniform(self)
        elif init == 'xavierNormal':
            weights_init_xavierNormal(self)
        elif init == 'xavierUniform':
            weights_init_xavierUniform(self)
        else:
            raise NotImplementedError("Invalid type of initialization!")
    def forward(self, x):
        # feed forward
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        l1 = F.max_pool2d(x, kernel_size=2, stride=2, padding=0) # /2
        l2 = F.max_pool2d(self.conv_block4(l1), kernel_size=2, stride=2, padding=0) # /4
        l3 = F.max_pool2d(self.conv_block5(l2), kernel_size=2, stride=2, padding=0) # /8
        x = self.conv_block6(l3) # /32
        g = self.dense(x) # batch_sizex512x1x1
        # pay attention
        if self.attention:
            c1, g1 = self.attn1(self.projector(l1), g)
            c2, g2 = self.attn2(l2, g)
            c3, g3 = self.attn3(l3, g)
            g = torch.cat((g1,g2,g3), dim=1) # batch_sizexC
            # classification layer
            x = self.classify(g) # batch_sizexnum_classes
        else:
            c1, c2, c3 = None, None, None
            x = self.classify(torch.squeeze(g))
        return [x, c1, c2, c3]


#@title Other
def worker_init_func(offset):

  def _init_fn(worker_id):
    random.seed(BASE_SEED + worker_id + offset)

  return _init_fn

#@title Transforms

transform_train = transforms.Compose([
    transforms.RandomCrop(IM_SIZE, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
])


#@title Load network
if OPT.EXAMPLE_TYPE not in EXAMPLE_TYPES:
  raise InvalidArgumentError(f"EXAMPLE_TYPE value '{OPT.EXAMPLE_TYPE}' unknown; options are {EXAMPLE_TYPES.keys()}.")
draw_func = EXAMPLE_TYPES[OPT.EXAMPLE_TYPE]

print('\nloading the dataset ...\n')
train_set = DrawDataset(draw_func, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=OPT.BATCH_SIZE, shuffle=True, num_workers=8, worker_init_fn=worker_init_func(0))

test_set = DrawDataset(draw_func, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=DATASET_SIZE, shuffle=False, num_workers=5, worker_init_fn=worker_init_func(9999))
print('done')


print('\nloading the network ...\n')
net = AttnVGG_after(im_size=IM_SIZE, num_classes=100, ## TODO 100??
    attention=OPT.USE_ATTN, normalize_attn=OPT.NORMALIZE_ATTN, init='xavierUniform')
criterion = nn.CrossEntropyLoss()
print('done')

### Skip moving to GPU
model = net


### optimizer
optimizer = optim.SGD(model.parameters(), lr=OPT.LR, momentum=0.9, weight_decay=5e-4)
lr_lambda = lambda epoch : np.power(0.5, int(epoch/25))
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

#@title Train
print('\nstart training ...\n')
step = 0
running_avg_accuracy = 0

writer = SummaryWriter(OPT.LOG_DIR)

for epoch in range(OPT.EPOCHS):
    images_disp = []
    # adjust learning rate
    scheduler.step()
    writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
    print("\nepoch %d learning rate %f\n" % (epoch, optimizer.param_groups[0]['lr']))
    # run for one epoch
    for aug in range(NUM_AUG):
        for i, data in enumerate(train_loader):
            # warm up
            model.train()   ## apply prev grad step (I think?)
            model.zero_grad()  ## set gradients to 0 bc we dont want to mix gradients b/w minibactches
            optimizer.zero_grad()
            inputs, labels = data
            if (aug == 0) and (i == 0): ## archive images in order to save to logs
                images_disp.append(inputs[0:36,:,:,:])
            # forward
            pred, __, __, __ = model(inputs)
            # backward
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
            # display results
            if i % 10 == 0:
                model.eval()  ## sets model in eval mode
                pred, __, __, __ = model(inputs)  ## changed since last run bc optimizer updates model
                predict = torch.argmax(pred, dim=1)
                total = labels.size(0)
                correct = torch.eq(predict, labels).sum().double().item()
                accuracy = correct / total
                running_avg_accuracy = 0.9*running_avg_accuracy + 0.1*accuracy
                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/accuracy', accuracy, step)
                writer.add_scalar('train/running_avg_accuracy', running_avg_accuracy, step)
                print("[epoch %d][aug %d/%d][%d/%d] loss %.4f accuracy %.2f%% running avg accuracy %.2f%%"
                    % (epoch, aug, NUM_AUG-1, i, len(train_loader)-1, loss.item(), (100*accuracy), (100*running_avg_accuracy)))
            step += 1

    # the end of each epoch: test & log
    print('\none epoch done, saving records ...\n')
    torch.save(model.state_dict(), os.path.join(OPT.LOG_DIR, 'net.pth'))
    if epoch == OPT.EPOCHS / 2:
        torch.save(model.state_dict(), os.path.join(OPT.LOG_DIR, 'net%d.pth' % epoch))
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():  ## saves comp by disabling backprop
        # log scalars for test set
        for i, data in enumerate(test_loader, 0):
            images_test, labels_test = data
            # images_test, labels_test = images_test.to(device), labels_test.to(device)
            if i == 0: ## archive images in order to save to logs
                images_disp.append(inputs[0:36,:,:,:])
            pred_test, __, __, __ = model(images_test)
            predict = torch.argmax(pred_test, 1)
            total += labels_test.size(0)
            correct += torch.eq(predict, labels_test).sum().double().item()
        writer.add_scalar('test/accuracy', correct/total, epoch)
        print("\n[epoch %d] accuracy on test data: %.2f%%\n" % (epoch, 100*correct/total))
        # log images
        if OPT.LOG_IMAGES:
            print('\nlog images ...\n')
            I_train = utils.make_grid(images_disp[0], nrow=6, normalize=True, scale_each=True, pad_value=1)
            writer.add_image('train/image', I_train, epoch)
            if epoch == 0:
                I_test = utils.make_grid(images_disp[1], nrow=6, normalize=True, scale_each=True, pad_value=1)
                writer.add_image('test/image', I_test, epoch)
        if OPT.LOG_IMAGES and OPT.USE_ATTN:
            print('\nlog attention maps ...\n')
            # # base factor
            # if opt.attn_mode == 'before':
            #     min_up_factor = 1
            # else:
            min_up_factor = 2
            # sigmoid or softmax
            if OPT.NORMALIZE_ATTN:
                vis_fun = visualize_attn_softmax
            else:
                vis_fun = visualize_attn_sigmoid
            # training data
            __, c1, c2, c3 = model(images_disp[0])
            if c1 is not None:
                attn1 = vis_fun(I_train, c1, up_factor=min_up_factor, nrow=6)
                writer.add_image('train/attention_map_1', attn1, epoch)
            if c2 is not None:
                attn2 = vis_fun(I_train, c2, up_factor=min_up_factor*2, nrow=6)
                writer.add_image('train/attention_map_2', attn2, epoch)
            if c3 is not None:
                attn3 = vis_fun(I_train, c3, up_factor=min_up_factor*4, nrow=6)
                writer.add_image('train/attention_map_3', attn3, epoch)
            # test data
            __, c1, c2, c3 = model(images_disp[1])
            if c1 is not None:
                attn1 = vis_fun(I_test, c1, up_factor=min_up_factor, nrow=6)
                writer.add_image('test/attention_map_1', attn1, epoch)
            if c2 is not None:
                attn2 = vis_fun(I_test, c2, up_factor=min_up_factor*2, nrow=6)
                writer.add_image('test/attention_map_2', attn2, epoch)
            if c3 is not None:
                attn3 = vis_fun(I_test, c3, up_factor=min_up_factor*4, nrow=6)
                writer.add_image('test/attention_map_3', attn3, epoch)