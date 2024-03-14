import argparse
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import GaussianDiffusion
from models.diffsf import DiffSF
from torch.utils.tensorboard import SummaryWriter
from utils.logger import Logger
import torch
import numpy as np
from tqdm.auto import tqdm
import math
import copy
from pathlib import Path
from torch.optim import AdamW
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from torchvision import transforms as utils
from datasetloader.flyingthings3d import FlyingThings3D_subset, FlyingThings3D_flownet3d
from datasetloader.kitti import KITTI_hplflownet, KITTI_flownet3d
from datasetloader.waymo import Waymo
from datasetloader.datasets import build_train_dataset, build_test_dataset

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        train_batch_size = 24,
        gradient_accumulate_every = 1,
        train_lr = 4e-4,
        train_num_steps = 600000,
        save_and_sample_every = 1000,
        results_folder = './results',
        checkpoint_dir = './checkpoints',
        max_grad_norm = 1.,
        dataset = 'f3d_occ'
    ):
        super().__init__()

        # model
        self.model = diffusion_model # diffusion model
        self.model_without_ddp = copy.deepcopy(self.model)
        # multiple GPUs
        if torch.cuda.device_count() > 1:
            print('Use %d GPUs' % torch.cuda.device_count())
            self.model.model = torch.nn.DataParallel(self.model.model)
            self.model_without_ddp.model = self.model.model.module
        else:
            self.model_without_ddp.model = self.model.model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # sampling and training hyperparameters
        self.save_and_sample_every = save_and_sample_every # 1000
        self.batch_size = train_batch_size # 24
        self.gradient_accumulate_every = gradient_accumulate_every # 1
        self.train_num_steps = train_num_steps # 600000
        self.max_grad_norm = max_grad_norm
        self.dataset = dataset

        # load training datasets
        self.train_dataset = build_train_dataset(self.dataset)
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                        shuffle=True, num_workers=cpu_count(),
                                                        pin_memory=True, drop_last=True,
                                                        sampler=None)
        # result and checkpoint folders
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok = True)

        # optimizer
        self.opt = AdamW(self.model_without_ddp.parameters(), lr=train_lr, weight_decay=1e-4)

        # step counter state
        self.step = 0

        # resume
        milestone = False
        if milestone:
            self.load(milestone)
            print('start_step: %d' % (self.step))
        last_epoch = self.step if milestone and self.step > 0 else -1

        # lr_scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.opt, train_lr,
            train_num_steps + 10,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy='cos',
            last_epoch=last_epoch,
        )

        # tensorboard
        summary_writer = SummaryWriter(checkpoint_dir)
        self.logger = Logger(self.lr_scheduler, summary_writer, summary_freq=500, start_step=self.step)


    @property
    def save(self, milestone):

        data = {
            'step': self.step,
            'model': self.get_state_dict(self.model_without_ddp),
            'opt': self.opt.state_dict(),
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        device = self.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)
        self.model_without_ddp.load_state_dict(data['model'], strict = True)
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])

        if 'version' in data:
            print(f"loading from version {data['version']}")

    def train(self):
        device = self.device

        epoch = 0
        while self.step < self.train_num_steps:
            model.train()
            metrics_3d = {}

            for i, sample in enumerate(self.train_loader):
                pcs = sample['pcs'].to(device)
                pcs = torch.permute(pcs, (0, 2, 1))
                flow_3d = sample['flow_3d'].to(device)

                total_loss = 0.
                loss, metrics_3d = self.model(flow_3d, pcs)
                if isinstance(loss, float):
                    continue
                if torch.isnan(loss):
                    continue
                loss = loss / self.gradient_accumulate_every
                total_loss += loss.item()
                metrics_3d.update({'total_loss': total_loss})
                # more efficient zero_grad
                for param in self.model_without_ddp.parameters():
                    param.grad = None
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()
                self.lr_scheduler.step()
                self.logger.push(metrics_3d)
                self.step += 1
                if self.step != 0 and (self.step % self.save_and_sample_every) == 0:
                    print('Save checkpoint at step: %d' % self.step)
                    milestone = self.step // self.save_and_sample_every
                    checkpoint_path = str(self.results_folder / f'model-{milestone}.pt')
                    torch.save({
                        'model': self.model_without_ddp.state_dict(),
                        'opt': self.opt.state_dict(),
                        'step': self.step,
                    }, checkpoint_path)
                        
                if self.step >= self.train_num_steps:
                    print('Training done')
                    return
            epoch += 1

class Tester(object):
    def __init__(
        self,
        diffusion_model,
        test_batch_size = 12,
        test_epoch = 600,
        results_folder = './results',
        dataset = 'kitti_occ'
    ):
        super().__init__()

        # model
        self.model = diffusion_model # diffusion model
        self.model_without_ddp = copy.deepcopy(self.model)
        
        # multiple GPUs
        if torch.cuda.device_count() > 1:
            print('Use %d GPUs' % torch.cuda.device_count())
            self.model.model = torch.nn.DataParallel(self.model.model)
            self.model_without_ddp.model = self.model.model.module
        else:
            self.model_without_ddp.model = self.model.model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # sampling and testing hyperparameters
        self.test_batch_size = test_batch_size
        self.test_epoch = test_epoch
        self.dataset = dataset

        # for logging results in a folder periodically
        self.results_folder = Path(results_folder)


    @property

    def load(self, milestone):
        device = self.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)
        self.model_without_ddp.load_state_dict(data['model'], strict = True)
        self.step = data['step']

        if 'version' in data:
            print(f"loading from version {data['version']}")

    def test(self):
        device = self.device
        self.model_without_ddp.eval()
        
        with torch.inference_mode():
            milestone = self.test_epoch
            #self.load(milestone)
            data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)
            self.model_without_ddp.load_state_dict(data['model'], strict = True)
            self.step = data['step']
            
            # load datasets
            self.val_dataset = build_test_dataset(self.dataset)
            self.val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.test_batch_size,
                                                            shuffle=True, num_workers=cpu_count(),
                                                            pin_memory=True, drop_last=True,
                                                            sampler=None)
            results = {}
            metrics_3d = {'counts': 0, 'EPE3d': 0.0, '5cm': 0.0, '10cm': 0.0, 'outlier': 0.0}
            metrics_3d_noc = {'counts': 0, 'EPE3d': 0.0, '5cm': 0.0, '10cm': 0.0, 'outlier': 0.0}

            for i, data in enumerate(self.val_loader):
                flow_preds = []
                pcs = data['pcs'].to(device) # 6*8192
                flow_3d = data['flow_3d'].to(device) # B 3 N
                flow_3d_mask = torch.ones(flow_3d[:,0,:].shape, dtype=torch.int64).cuda()
                if self.dataset == 'kitti_occ' or self.dataset == 'kitti_nonocc': # rescale kitti dataset to match f3d dataset
                    occ_mask_3d = torch.zeros(flow_3d[:,0,:].shape, dtype=torch.int64).cuda()
                    pcs = torch.permute(pcs, (0, 2, 1)) * torch.tensor([.5, 1., .5, .5, 1., .5])[None,:,None].cuda()
                    flow_pred = self.model_without_ddp.sample(pcs, return_all_timesteps = False) / torch.tensor([.5, 1., .5])[None,:,None].cuda()
                else:
                    occ_mask_3d = data['occ_mask_3d'].to(device) # B N
                    pcs = torch.permute(pcs, (0, 2, 1))
                    flow_pred = self.model_without_ddp.sample(pcs, return_all_timesteps = False) # B 3 N

                #flow_preds.append(flow_pred)
                #for j in range (19):
                #    flow_pred = self.model_without_ddp.sample(pcs, return_all_timesteps = False) # B 3 N
                #    flow_preds.append(flow_pred)
                #stack_preds = torch.stack(flow_preds)
                ##flow_pred, _ = torch.median(stack_preds, dim = 0)
                #flow_pred = torch.mean(stack_preds, dim = 0)
                #flow_pred_std = torch.std(stack_preds, dim = 0)

                # save testing images
#                import numpy as np
#                np.save('results/things_occ/'+format(i, '04d')+'pcs', pcs.cpu())
#                np.save('results/things_occ/'+format(i, '04d')+'_flow_3d_pred', flow_pred.cpu())
#                np.save('results/things_occ/'+format(i, '04d')+'_flow_3d_target', flow_3d.cpu())
#                np.save('results/things_occ/'+format(i, '04d')+'_flow_3d_pred_std', flow_pred_std.cpu())
#                np.save('results/things_occ/'+format(i, '04d')+'_stack_preds', stack_preds.cpu())

                epe3d_map = torch.sqrt(torch.sum((flow_pred - flow_3d) ** 2, dim=1)) # B N
                # evaluate on occluded points
                flow_3d_mask = torch.logical_and(flow_3d_mask, torch.logical_not(torch.isnan(epe3d_map)))
                metrics_3d['counts'] += epe3d_map[flow_3d_mask].shape[0]
                metrics_3d['EPE3d'] += epe3d_map[flow_3d_mask].sum().item()
                metrics_3d['5cm'] += torch.count_nonzero(torch.logical_or((epe3d_map[flow_3d_mask] < 0.05), (epe3d_map[flow_3d_mask]/torch.sqrt(torch.sum((flow_3d) ** 2, dim=1))[flow_3d_mask] < 0.05))).item()
                metrics_3d['10cm'] += torch.count_nonzero(torch.logical_or((epe3d_map[flow_3d_mask] < 0.1), (epe3d_map[flow_3d_mask]/torch.sqrt(torch.sum((flow_3d) ** 2, dim=1))[flow_3d_mask] < 0.1))).item()
                metrics_3d['outlier'] += torch.count_nonzero(torch.logical_or((epe3d_map[flow_3d_mask] > 0.3), (epe3d_map[flow_3d_mask]/torch.sqrt(torch.sum((flow_3d) ** 2, dim=1))[flow_3d_mask] > 0.1))).item()

                # evaluate on non-occluded points
                flow_3d_mask = torch.logical_and(occ_mask_3d == 0, flow_3d_mask)
                epe3d_map_noc = epe3d_map[flow_3d_mask]
                metrics_3d_noc['counts'] += epe3d_map_noc.shape[0]
                metrics_3d_noc['EPE3d'] += epe3d_map_noc.sum().item()
                metrics_3d_noc['5cm'] += torch.count_nonzero(torch.logical_or((epe3d_map_noc < 0.05), (epe3d_map_noc/torch.sqrt(torch.sum((flow_3d) ** 2, dim=1))[flow_3d_mask] < 0.05))).item()
                metrics_3d_noc['10cm'] += torch.count_nonzero(torch.logical_or((epe3d_map_noc < 0.1), (epe3d_map_noc/torch.sqrt(torch.sum((flow_3d) ** 2, dim=1))[flow_3d_mask] < 0.1))).item()
                metrics_3d_noc['outlier'] += torch.count_nonzero(torch.logical_or((epe3d_map_noc > 0.3), (epe3d_map_noc/torch.sqrt(torch.sum((flow_3d) ** 2, dim=1))[flow_3d_mask] > 0.1))).item()

            print('#### 3D Metrics ####')
            results['EPE'] = metrics_3d['EPE3d'] / metrics_3d['counts']
            results['5cm'] = metrics_3d['5cm'] / metrics_3d['counts'] * 100.0
            results['10cm'] = metrics_3d['10cm'] / metrics_3d['counts'] * 100.0
            results['outlier'] = metrics_3d['outlier'] / metrics_3d['counts'] * 100.0
            print("Validation Things EPE: %.6f, 5cm: %.4f, 10cm: %.4f, outlier: %.6f" % (results['EPE'], results['5cm'], results['10cm'], results['outlier']))

            print('#### 3D Metrics non-occluded ####')
            results['EPE_non-occluded'] = metrics_3d_noc['EPE3d'] / metrics_3d_noc['counts']
            results['5cm_non-occluded'] = metrics_3d_noc['5cm'] / metrics_3d_noc['counts'] * 100.0
            results['10cm_non-occluded'] = metrics_3d_noc['10cm'] / metrics_3d_noc['counts'] * 100.0
            results['outlier_non-occluded'] = metrics_3d_noc['outlier'] / metrics_3d_noc['counts'] * 100.0
            print("Validation Things EPE: %.6f, 5cm: %.4f, 10cm: %.4f, outlier: %.6f" % (results['EPE_non-occluded'], results['5cm_non-occluded'], results['10cm_non-occluded'], results['outlier_non-occluded']))


def get_args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--checkpoint_dir', default='./checkpoints', type=str, help='where to save the training log and models')
    parser.add_argument('--result_dir', default='./results', type=str, help='where to save the training log and models')
    parser.add_argument('--train_dataset', default='f3d_occ', type=str, help='training dataset on different datasets (f3d_occ / f3d_nonocc / waymo)')
    parser.add_argument('--val_dataset', default='f3d_occ', type=str, help='validation dataset on different datasets (f3d_occ / f3d_nonocc / kitti_occ / kitti_nonocc / waymo)')
    # training
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--train_batch_size', default=24, type=int)
    parser.add_argument('--test_batch_size', default=12, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--seed', default=326, type=int)
    # resume pretrained model or resume training
    parser.add_argument('--resume', default=None, type=int,
                        help='resume from pretrained model or resume from unexpectedly terminated training')
    # model: learnable parameters
    parser.add_argument('--timesteps', default=20, type=int)
    parser.add_argument('--samplingtimesteps', default=2, type=int)
    parser.add_argument('--feature_channels', default=128, type=int)
    parser.add_argument('--backbone', default='DGCNN', type=str, help='feature extraction backbone (DGCNN / PointNet / MLP)')
    parser.add_argument('--ffn_dim_expansion', default=4, type=int)
    parser.add_argument('--num_transformer_pt_layers', default=1, type=int)
    parser.add_argument('--num_transformer_layers', default=14, type=int)
    # evaluation
    parser.add_argument('--eval', action='store_true',
                        help='evaluation after training done')
    # log
    parser.add_argument('--summary_freq', default=500, type=int, help='Summary frequency to tensorboard (iterations)')
    parser.add_argument('--save_ckpt_freq', default=1000, type=int, help='Save checkpoint frequency (steps)')
    parser.add_argument('--num_steps', default=600000, type=int)
    parser.add_argument('--test_epoch', default=600, type=int)
    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    # set random seed
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model
    model = DiffSF(backbone=args.backbone,
                channels=args.feature_channels,
                ffn_dim_expansion=args.ffn_dim_expansion,
                num_transformer_pt_layers=args.num_transformer_pt_layers,
                num_transformer_layers=args.num_transformer_layers).to(device)
    # diffusion
    diffusion = GaussianDiffusion(
        model,
        objective = 'pred_x0', # pred_x0
        beta_schedule = 'cosine',  # sigmoid, cosine, linear
        timesteps = args.timesteps,           # number of steps
        sampling_timesteps = args.samplingtimesteps    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    ).to(device)

    # test
    if args.eval:
        tester = Tester(
            diffusion,
            test_batch_size = args.test_batch_size,
            test_epoch = args.test_epoch,                # total training steps
            results_folder = args.result_dir,
            dataset = args.train_dataset     # f3d_occ / f3d_nonocc / kitti_occ / kitti_nonocc / waymo
        )
        tester.test()

    # train
    else:
        trainer = Trainer(
            diffusion,
            train_batch_size = args.train_batch_size,
            train_lr = args.lr,
            train_num_steps = args.num_steps,         # total training steps
            results_folder = args.result_dir,
            checkpoint_dir = args.checkpoint_dir,
            dataset = args.val_dataset                # f3d_occ / f3d_nonocc / waymo
        )
        trainer.train()