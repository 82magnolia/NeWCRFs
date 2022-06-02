from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import os, sys, errno
import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import post_process_depth, flip_lr
from networks.NewCRFDepth import NewCRFDepth


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='NeWCRFs PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name', type=str, help='model name', default='newcrfs')
parser.add_argument('--encoder', type=str, help='type of encoder, base07, large07', default='large07')
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--dataset', type=str, help='dataset to train on', default='nyu')
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--save_viz', help='if set, save visulization of the outputs', action='store_true')

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.dataset == 'kitti' or args.dataset == 'nyu':
    from dataloaders.dataloader import NewDataLoader
elif args.dataset == 'kittipred':
    from dataloaders.dataloader_kittipred import NewDataLoader

model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def test(params):
    """Test function."""
    args.mode = 'test'
    dataloader = NewDataLoader(args, 'test')
    
    model = NewCRFDepth(version='large07', inv_depth=False, max_depth=args.max_depth)
    model = torch.nn.DataParallel(model)
    
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_test_samples = get_num_lines(args.filenames_file)

    with open(args.filenames_file) as f:
        lines = f.readlines()

    print('now testing {} files with {}'.format(num_test_samples, args.checkpoint_path))

    save_name = 'models/' + args.filenames_file.split('/')[-1].replace('.txt', '')
    if not os.path.exists(save_name):
        try:
            os.mkdir(save_name)
            os.mkdir(save_name + '/raw')
            if args.save_viz:
                os.mkdir(save_name + '/cmap')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    start_time = time.time()
    with torch.no_grad():
        for s, sample in enumerate(tqdm(dataloader.data)):
            image = Variable(sample['image'].cuda())
            B, C, H, W = image.shape
            new_image = torch.zeros([B, C, args.input_height, args.input_width])
            adj_height = min(args.input_height, H)
            adj_width = min(args.input_width, W)
            new_image[..., args.input_height // 2 - adj_height // 2: args.input_height // 2 + adj_height // 2, args.input_width // 2 - adj_width // 2: args.input_width // 2 + adj_width // 2] = image[..., H // 2 - adj_height // 2: H // 2 + adj_height // 2, W // 2 - adj_width // 2: W // 2 + adj_width // 2]
            depth_est = model(new_image)
            post_process = True
            if post_process:
                image_flipped = flip_lr(new_image)
                depth_est_flipped = model(image_flipped)
                depth_est = post_process_depth(depth_est, depth_est_flipped)

            pred_depth = depth_est.cpu().numpy().squeeze()

            if args.do_kb_crop:
                height, width = 352, 1216
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
                pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
                pred_depth = pred_depth_uncropped
            
            filename_pred_png = save_name + '/raw/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
            pred_depth_scaled = pred_depth * 1000.0
            pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
            cv2.imwrite(filename_pred_png, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            if args.save_viz:
                filename_cmap_png = save_name + '/cmap/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
                plt.imsave(filename_cmap_png, (10 - pred_depth) / 10, cmap='plasma')


    elapsed_time = time.time() - start_time
    print('Elapesed time: %s' % str(elapsed_time))
    print('Done.')
    
    return


if __name__ == '__main__':
    test(args)
