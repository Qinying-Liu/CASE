import numpy as np
import argparse
import os


def parse_args():
    description = 'Weakly supervised action localization'
    parser = argparse.ArgumentParser(description=description)

    # dataset parameters
    parser.add_argument('--data_path', type=str, default='data/THUMOS14')
    parser.add_argument('--exp_name', type=str, required=True, help="Name of the current experiment")
    parser.add_argument('--output_dir', type=str, default='./outputs')

    # data parameters
    parser.add_argument('--modal', type=str, default='all', choices=['rgb', 'flow', 'all'])
    parser.add_argument('--num_segments', default=750, type=int)
    parser.add_argument('--scale', default=24, type=int)

    # model parameters
    parser.add_argument('--model_name', type=str, default='ThumosModel', help="Which model to use")

    # training parameter
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rates for steps(list form)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--detection_inf_step', default=50, type=int, help="Run detection inference every n steps")
    parser.add_argument('--q_val', default=0.7, type=float)

    # inference parameters
    parser.add_argument('--inference_only', action='store_true', default=False)
    parser.add_argument('--class_th', type=float, default=0.25)
    parser.add_argument('--model_file', type=str, default=None, help='the path of pre-trained model file')
    parser.add_argument('--gamma', type=float, default=0.15, help='Gamma for oic class confidence')
    parser.add_argument('--soft_nms', default=True, action='store_true')
    parser.add_argument('--nms_alpha', default=0.35, type=float)
    parser.add_argument('--nms_thresh', default=0.4, type=float)
    parser.add_argument('--load_weight', default=False, action='store_true')

    # system parameters
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=1, help='random seed (-1 for no manual seed)')
    parser.add_argument('--verbose', default=False, action='store_true')

    # CASE parameters
    parser.add_argument('--num_clusters', default=16, type=int)
    parser.add_argument('--temp', default=10., type=float)
    parser.add_argument('--std', default=10., type=float)
    parser.add_argument('--w_clu', default=1., type=float)
    parser.add_argument('--w_cls', default=0.3, type=float)

    return init_args(parser.parse_args())


def init_args(args):
    args.model_path = os.path.join(args.output_dir, args.exp_name)
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    args.model_path = os.path.join(args.output_dir, args.exp_name)

    return args


class Config(object):
    def __init__(self, args):
        self.lr = args.lr
        self.num_classes = 20
        self.modal = args.modal
        if self.modal == 'all':
            self.len_feature = 2048
        else:
            self.len_feature = 1024
        self.batch_size = args.batch_size
        self.data_path = args.data_path
        self.model_path = os.path.join(args.output_dir, args.exp_name)
        self.num_workers = args.num_workers
        self.class_thresh = args.class_th
        self.act_thresh = np.arange(0.1, 1.0, 0.1)
        self.q_val = args.q_val
        self.scale = args.scale
        self.gt_path = os.path.join(self.data_path, 'gt.json')
        self.model_file = args.model_file
        self.seed = args.seed
        self.feature_fps = 25
        self.num_segments = args.num_segments
        self.num_epochs = args.num_epochs
        self.gamma = args.gamma
        self.inference_only = args.inference_only
        self.model_name = args.model_name
        self.detection_inf_step = args.detection_inf_step
        self.soft_nms = args.soft_nms
        self.nms_alpha = args.nms_alpha
        self.nms_thresh = args.nms_thresh
        self.load_weight = args.load_weight
        self.verbose = args.verbose

        self.num_clusters = args.num_clusters
        self.temp = args.temp
        self.std = args.std
        self.w_clu = args.w_clu
        self.w_cls = args.w_cls
