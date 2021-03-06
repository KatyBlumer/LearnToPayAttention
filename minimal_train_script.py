import os
import sys
sys.path.append(f"{os.getcwd()}/venv_t/lib/python3.7/site-packages")

import argparse
import collections
import cv2
import numpy as np
import random
import subprocess

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

parser.add_argument("--RUN_MULTI", action='store_true', help='if True, rerun this script with multiple example types')

parser.add_argument("--BATCH_SIZE", type=int, default=32, help="batch size")
parser.add_argument("--EPOCHS", type=int, default=10, help="number of epochs")
parser.add_argument("--LR", type=float, default=0.01, help="initial learning rate")
parser.add_argument("--LOG_DIR", type=str, default="logs", help='path of log files')

parser.add_argument("--NORMALIZE_ATTN", action='store_true', help='if True, attention map is normalized by softmax; otherwise use sigmoid')
parser.add_argument("--USE_ATTN", action='store_false', help='turn down attention')
parser.add_argument("--LOG_IMAGES", action='store_false', help='log images and (is available) attention maps')

parser.add_argument("--EXAMPLE_TYPE", type=str, nargs="*", help='Which type of task to train on')

OPT = parser.parse_args()

BASE_SEED = 0

# LOG_DIR = "logs/TEMP_CHANGE_THIS"

DATASET_SIZE = 1000
# BATCH_SIZE = 128
# EPOCHS = 300
# LR = 0.1

# NORMALIZE_ATTN = False
# USE_ATTN = True
# LOG_IMAGES = True  # False

NUM_AUG = 1
IM_SIZE = 32

STEPS_PER_LOG = 2

GRID_BORDER_VALUE = 0.9

#@title Attn visualizers
def visualize_attn_base(I, c, up_factor, nrow, heatmap_func, norm_grid):
  # image
  img = I.permute((1,2,0)).cpu().numpy()
  # compute the heatmap
  a = heatmap_func(c)
  if up_factor > 1:
      a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
  attn = utils.make_grid(a, nrow=nrow, pad_value=GRID_BORDER_VALUE, normalize=norm_grid, scale_each=norm_grid)
  attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
  attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
  attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
  attn = np.float32(attn) / 255
  # add the heatmap to the image
  vis = 0.6 * img + 0.4 * attn
  return torch.from_numpy(vis).permute(2,0,1)

def visualize_attn_softmax(I, c, up_factor, nrow):
    def softmax_heatmap_func(c):
      N,C,W,H = c.size()
      a = F.softmax(c.view(N,C,-1), dim=2).view(N,C,W,H)
    return visualize_attn_base(I, c, up_factor, nrow, softmax_heatmap_func, norm_grid=True)

def visualize_attn_sigmoid(I, c, up_factor, nrow):
  return visualize_attn_base(I, c, up_factor, nrow, heatmap_func=torch.sigmoid, norm_grid=False)


def scale_batch(batch):
  # WARNING assumes N,C,W,H dims
  batch  = batch - batch.min(axis=(1,2,3))[:, np.newaxis, np.newaxis, np.newaxis]
  batch =  batch / batch.max(axis=(1,2,3))[:, np.newaxis, np.newaxis, np.newaxis]
  n = int(32 / batch.shape[-1])
  batch = np.squeeze(
      np.kron(
          batch, np.ones((1, 1, n, n))
      )
  )
  return batch

def visualize_attn_composite(I, attn_maps, nrow):
  img = I.permute((1,2,0)).cpu().numpy()
  attn = torch.from_numpy(
      np.stack(
          (
            scale_batch(attn_maps[0].cpu().detach().numpy()),
            scale_batch(attn_maps[1].cpu().detach().numpy()),
            scale_batch(attn_maps[2].cpu().detach().numpy())
          ),
          axis=1
          )
      )
  attn = utils.make_grid(attn, nrow=nrow, pad_value=GRID_BORDER_VALUE)
  attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
  attn = np.float32(attn) / 255
  # add the heatmap to the image
  vis = 0.5 * img + 0.5  * attn
  return torch.from_numpy(vis).permute(2,0,1)


#@title Dataset drawing
def create_rgb(r, g, b):
  return np.dstack(
      (r * 255,
       g * 255,
       b * 255
       )
  ).astype(np.uint8)

def create_rgb_randcolor(r, g, b):
  return np.dstack(
      (r * (random.random() * 255),
       g * (random.random() * 255),
       b * (random.random() * 255)
       )
  ).astype(np.uint8)

def create_randcolor(shape):
  return create_rgb_randcolor(shape, shape, shape)


def add_to_im(lower, upper, bg_val=0):
  if len(upper.shape) == 2:
    upper = np.expand_dims(upper, axis=2)

  # Discard pixels where the object exists, so we can replace them
  keep_pixels = (upper == bg_val).all(axis=2)
  keep_pixels = np.expand_dims(keep_pixels, axis=2)
  lower = lower * keep_pixels

  return lower + upper


def random_background():
  return create_rgb(
      np.random.rand(32, 32),
      np.random.rand(32, 32),
      np.random.rand(32, 32)
  )


def add_background(im, bg_val=0):
  im = add_to_im(lower=random_background(), upper=im, bg_val=bg_val)
  return im

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


def draw_square(center, rad=2, thickness=2, imsize=32):
  xx, yy = np.mgrid[:imsize, :imsize]
  center_x, center_y = center
  xx = np.abs(xx - center_x)
  yy = np.abs(yy - center_y)

  box = np.logical_xor(
      np.logical_and(xx < (rad + thickness), yy < (rad + thickness)),
      np.logical_and(xx < rad, yy < rad),
  )
  return box

def random_square(imsize):
  center_x = random.randrange(3, imsize - 3)
  center_y = random.randrange(3, imsize - 3)
  return draw_square([center_x, center_y])


def create_presence_example(label, imsize):
  if label:
    circ = random_circle(imsize)
  else:
    circ = np.zeros([imsize, imsize])

  return add_background(create_randcolor(circ))

def create_color_example(label, imsize):
  circ = random_circle(imsize)
  zeros = np.zeros([imsize, imsize])

  if label:
    return add_background(create_rgb_randcolor(circ, zeros, zeros))
  else:
    return add_background(create_rgb_randcolor(zeros, circ, zeros))


def create_shape_example(label, imsize):
  if label:
    shape = random_square(imsize)
  else:
    shape = random_circle(imsize)

  return add_background(create_randcolor(shape))


def create_number_example(label, imsize):
  circ = create_randcolor(random_circle(imsize))
  if label:
    circ = add_to_im(lower=circ, upper=create_randcolor(random_circle(imsize)))

  return add_background(circ)


def create_location_example(label, imsize):
  center_x = random.randrange(3, (imsize / 2) - 3)
  center_y = random.randrange(3, imsize - 3)

  if label:
    center_x += (imsize / 2)

  circ = draw_circle([center_x, center_y])
  return add_background(create_randcolor(circ))

def create_distance_example(label, imsize):
  dists = [5, 9]
  dist = dists[label]

  center_x = random.randrange(3, imsize - (3+max(dists)))
  center_y = random.randrange(3, imsize - (3+max(dists)))
  rad  = 7
  thickness = 4

  im = add_background(add_to_im(
      lower = create_randcolor(draw_circle([center_x, center_y])),
      upper = create_randcolor(draw_circle([center_x+dist, center_y+dist])),
      )
  )
  return im


EXAMPLE_TYPES = {
  'presence': create_presence_example,
  'color': create_color_example,
  'number': create_number_example,
  'shape': create_shape_example,
  'location': create_location_example,
  'distance': create_distance_example,
}


class DrawDataset(torch.utils.data.Dataset):

    def __init__(self, draw_func, transform=None, imsize=32):
        self.imsize=imsize
        self.transform = transform
        self.draw_func = draw_func

    def __len__(self):
        return DATASET_SIZE

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # print("Image idx", idx)

        label = random.randrange(2)

        im = self.draw_func(label, self.imsize)

        im = Image.fromarray(im)
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


def is_interesting(attn_batch):
  # any of the batch images, have variance above 0 (only 1 color channel)
  return (torch.var(attn_batch.view(attn_batch.shape[0], -1), dim=1) > 0).any()


def any_is_interesting(li):
  for attn_map in li:
    if is_interesting(attn_map):
      return True
  return False


def save_attn_map(attn, log_dir, label, is_training, step):
  train_or_test = "train" if is_training else "test"
  fname = f"{log_dir}/attn_maps/{train_or_test}_step{step:04}_map{label}.npy"
  attn = attn.detach().numpy()
  try:
    np.save(fname, attn, allow_pickle = False)
  except:
    os.makedirs(f"{log_dir}/attn_maps/")
    np.save(fname, attn, allow_pickle = False)


def save_if_interesting(li, step, is_training, log_dir):
  if not any_is_interesting(li):
    return False

  print("Saving raw attention maps")
  for i, attn_map in enumerate(li):
    save_attn_map(attn_map, log_dir, i, is_training, step)

  return True


def train_step(
  i, aug, epoch, step, train_loader_len,
  data, model, optimizer, criterion,
  writer
  ):
  # warm up
  model.train()   ## set to train mode (vs eval)
  model.zero_grad()  ## set gradients to 0 bc we dont want to mix gradients b/w minibatches
  optimizer.zero_grad()
  inputs, labels = data
  train_batch_disp = inputs[0:36,:,:,:]
  if (step == 0): # so we have something to look at right away
      I_train = utils.make_grid(train_batch_disp, nrow=6, pad_value=GRID_BORDER_VALUE)
      writer.add_image('train/image', I_train, epoch)
  # forward
  pred, __, __, __ = model(inputs)
  # backward
  loss = criterion(pred, labels)
  loss.backward()
  optimizer.step()
  # display train results
  if i % STEPS_PER_LOG == 0:
      writer.add_scalar('train/epoch', epoch + (i / train_loader_len), step)
      model.eval()  ## sets model in eval mode
      pred, __, __, __ = model(inputs)  ## changed since last run bc optimizer updates model
      predict = torch.argmax(pred, dim=1)
      total = labels.size(0)
      correct = torch.eq(predict, labels).sum().double().item()
      accuracy = correct / total
      writer.add_scalar('train/loss', loss.item(), step)
      writer.add_scalar('train/accuracy', accuracy, step)
      print(f"[epoch {epoch}][aug {aug}/{NUM_AUG-1}][{i}/{train_loader_len-1}] "
            f"loss {loss.item():.4f} accuracy {(100*accuracy):.2f}% "
            )
  return train_batch_disp


def maybe_log_images(step, train_batch_disp, test_loader, model, writer, log_dir):
  is_log_step = (step % STEPS_PER_LOG == 0)
  __, c1_train, c2_train, c3_train = model(train_batch_disp)

  for i, attn in enumerate([c1_train, c2_train, c3_train]):
    writer.add_scalar(f'train/var_attn_{i}', torch.var(attn), step)

  interesting = save_if_interesting([c1_train, c2_train, c3_train], step, True, log_dir)
  if not (interesting or is_log_step):
    return

  print('\nlog images ...\n')

  for images_test, labels_test in test_loader:
      test_batch_disp = images_test[0:36,:,:,:]
      if step == 0:
        np.save(f"{log_dir}/attn_maps/test_images.npy", test_batch_disp.detach().numpy(), allow_pickle = False)
      break  # should only run once anyway; test batch size = DATASET_SIZE

  I_train = utils.make_grid(train_batch_disp, nrow=6, pad_value=GRID_BORDER_VALUE)
  I_test = utils.make_grid(test_batch_disp, nrow=6, pad_value=GRID_BORDER_VALUE)
  if step == 0:
      writer.add_image('test/image', I_test, step)
  else:
      ## Only save after step 0, because for step 0 we did it in training loop
      writer.add_image('train/image', I_train, step)

  if OPT.USE_ATTN:
      __, c1_test, c2_test, c3_test = model(test_batch_disp)
      for i, attn in enumerate([c1_test, c2_test, c3_test]):
        writer.add_scalar(f'test/var_attn_{i}', torch.var(attn), step)
      save_if_interesting([c1_test, c2_test, c3_test], step, False, log_dir)
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
      if c1_train is not None:
          attn1 = vis_fun(I_train, c1_train, up_factor=min_up_factor, nrow=6)
          writer.add_image('train/attention_map_1', attn1, step)
      if c2_train is not None:
          attn2 = vis_fun(I_train, c2_train, up_factor=min_up_factor*2, nrow=6)
          writer.add_image('train/attention_map_2', attn2, step)
      if c3_train is not None:
          attn3 = vis_fun(I_train, c3_train, up_factor=min_up_factor*4, nrow=6)
          writer.add_image('train/attention_map_3', attn3, step)
      # test data
      if c1_test is not None:
          attn1 = vis_fun(I_test, c1_test, up_factor=min_up_factor, nrow=6)
          writer.add_image('test/attention_map_1', attn1, step)
      if c2_test is not None:
          attn2 = vis_fun(I_test, c2_test, up_factor=min_up_factor*2, nrow=6)
          writer.add_image('test/attention_map_2', attn2, step)
      if c3_test is not None:
          attn3 = vis_fun(I_test, c3_test, up_factor=min_up_factor*4, nrow=6)
          writer.add_image('test/attention_map_3', attn3, step)

      composite_train = visualize_attn_composite(I_train, [c1_train, c2_train, c3_train], nrow=6)
      writer.add_image('train/attention_composite', composite_train, step)
      composite_test = visualize_attn_composite(I_test, [c1_test, c2_test, c3_test], nrow=6)
      writer.add_image('test/attention_composite', composite_test, step)


def test_full_set(epoch, test_loader, model, writer):
  model.eval()
  total = 0
  correct = 0
  with torch.no_grad():  ## saves comp by disabling backprop
      # log scalars for test set
      for i, data in enumerate(test_loader, 0):  # should only run once; test batch size = DATASET_SIZE
          images_test, labels_test = data
          pred_test, __, __, __ = model(images_test)
          predict = torch.argmax(pred_test, 1)
          total += labels_test.size(0)
          correct += torch.eq(predict, labels_test).sum().double().item()
      writer.add_scalar('test/accuracy', correct/total, epoch)
      print("\n[epoch %d] accuracy on test data: %.2f%%\n" % (epoch, 100*correct/total))


def train(draw_func, log_dir):
  torch.backends.cudnn.deterministic = True
  torch.manual_seed(BASE_SEED)
  torch.cuda.manual_seed_all(BASE_SEED)

  #@title Transforms
  transform_train = transforms.Compose([
      # transforms.RandomCrop(IM_SIZE, padding=4, fill=(255, 255, 255)),
      # transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
  ])
  transform_test = transforms.Compose([
      transforms.ToTensor(),
      # transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
  ])


  #@title Load network
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

  writer = SummaryWriter(log_dir)

  for epoch in range(OPT.EPOCHS):
      images_disp = []
      writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
      print("\nepoch %d learning rate %f\n" % (epoch, optimizer.param_groups[0]['lr']))
      # run for one epoch
      for aug in range(NUM_AUG):
          for i, data in enumerate(train_loader):
              train_batch_disp = train_step(
                i, aug, epoch, step, len(train_loader),
                data, model, optimizer, criterion,
                writer)
              if OPT.LOG_IMAGES:
                maybe_log_images(step, train_batch_disp, test_loader, model, writer, log_dir)

              step += 1

      # adjust learning rate
      scheduler.step()

      # the end of each epoch: test & log
      print('\none epoch done, saving records ...\n')
      torch.save(model.state_dict(), os.path.join(log_dir, 'net.pth'))
      if epoch == OPT.EPOCHS / 2:
          torch.save(model.state_dict(), os.path.join(log_dir, 'net%d.pth' % epoch))

      test_full_set(epoch, test_loader, model, writer)

  return 0



if OPT.RUN_MULTI:
  if OPT.EXAMPLE_TYPE:
    filtered_types = {k: v for k, v in EXAMPLE_TYPES.items() if k in OPT.EXAMPLE_TYPE}
  else:
    filtered_types = EXAMPLE_TYPES

  procs = []
  for name in filtered_types.keys():
    log_subdir = f"{OPT.LOG_DIR}/{name}/"
    os.makedirs(log_subdir, exist_ok=True)
    cmd_l = ["python", "minimal_train_script.py", "--EXAMPLE_TYPE", f"{name}", "--LOG_DIR", log_subdir]
    print(f"\n_____________RUNNING {name}:\n{cmd_l}\n_____________________")
    with open(f"{log_subdir}/stdout.txt", 'w') as f:
      proc = subprocess.Popen(cmd_l, stdout=f)
    print(f"PID: {proc.pid}")
    procs.append(proc)

  print(f"_______________TO KILL: {'; '.join([f'kill {p.pid}' for p in procs])}")
  exit_codes = [proc.wait() for proc in procs]
  print(f"Exit codes: {exit_codes}")

else:
  ex_type = OPT.EXAMPLE_TYPE[0]
  if ex_type not in EXAMPLE_TYPES:
    raise ValueError(f"EXAMPLE_TYPE value '{OPT.EXAMPLE_TYPE}' unknown; options are {EXAMPLE_TYPES.keys()}; or leave blank to use all.")

  draw_func = EXAMPLE_TYPES[ex_type]
  train(draw_func, OPT.LOG_DIR)
