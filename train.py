import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import random
from PromptReg import PromptReg
from dataset import MultiTaskRegistrationDataset
import torch.nn.functional as nnf



class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)



class Grad3d(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad



class MIND_loss(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """
    def __init__(self, device, win=None):
        super(MIND_loss, self).__init__()
        self.win = win
        self.device = device

    def pdist_squared(self, x):
        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def MINDSSC(self, img, radius=2, dilation=2):
        # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor
        kernel_size = radius * 2 + 1

        six_neighbourhood = torch.Tensor([[0, 1, 1],
                                        [1, 1, 0],
                                        [1, 0, 1],
                                        [1, 1, 2],
                                        [2, 1, 1],
                                        [1, 2, 1]]).long()

        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
        mshift1 = torch.zeros(12, 1, 3, 3, 3).to(self.device)
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3).to(self.device)
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        rpad1 = torch.nn.ReplicationPad3d(dilation)
        rpad2 = torch.nn.ReplicationPad3d(radius)

        ssd = F.avg_pool3d(rpad2(
            (F.conv3d(rpad1(img), mshift1, dilation=dilation) - F.conv3d(rpad1(img), mshift2, dilation=dilation)) ** 2),
            kernel_size, stride=1)

        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item())
        mind = mind / mind_var
        mind = torch.exp(-mind)
        
        mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]
        
        return mind

    def forward(self, y_pred, y_true):
        return torch.mean((self.MINDSSC(y_pred) - self.MINDSSC(y_true)) ** 2)

def adjust_learning_rate_power(optimizer, epoch, max_epochs, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power( 1 - (epoch) / max_epochs, power), 8)


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    

def parse_args():
    parser = argparse.ArgumentParser(description='Training script for multi-task registration')
    parser.add_argument('--exclude-tasks', nargs='+', default=['Abdominal'], 
                      help='Tasks to exclude from training')
    parser.add_argument('--gpu', type=str, default='0',
                      help='GPU device ID to use')
    parser.add_argument('--batch-size', type=int, default=1,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=300,
                      help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0001,
                      help='Initial learning rate')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    return parser.parse_args()
def main():

    args = parse_args()
    set_seed(args.seed)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
  
    data_root = '/path/to/dataset'
    target_size = (160, 160, 160)
    batch_size = args.batch_size
    num_epochs = args.epochs
    initial_lr = args.lr

    save_dir = '/path/to/save/checkpoints'
   
    os.makedirs(save_dir, exist_ok=True)
    
    
    train_dataset = MultiTaskRegistrationDataset(
        data_root=data_root,
        target_size=target_size,
        split='train',
        exclude_tasks=args.exclude_tasks
    )
    train_loader = train_dataset.get_dataloader(batch_size=batch_size)
    
   
    test_datasets = {
        'Abdominal': ABDODataset(os.path.join(data_root, 'ABDO'), split='test'),
        'Brain': BrainDataset(os.path.join(data_root, 'Brain'), split='test'),
        'Hippocampus': HaimaDataset(os.path.join(data_root, 'Haima'), split='test'),
        'Cardiac': HeartDataset(os.path.join(data_root, 'Heart'), split='test'),
        'Hip': HipDataset(os.path.join(data_root, 'Hip'), split='test')
    }
    

    model = PromptReg(task_total_number=5)
    model.to(device)

    mind_loss = MIND_loss(device)
    grad_loss = Grad3d(penalty='l2')
  
    optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=0, amsgrad=True)
    
    for epoch in range(num_epochs):
        current_lr = adjust_learning_rate_power(optimizer, epoch, num_epochs, init_lr=initial_lr, power=0.9)
        
        model.train()
        epoch_loss = 0
        start_time = time.time()
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            moving = batch['moving'].to(device)
            fixed = batch['fixed'].to(device)
            task_id = batch['task_id'].to(device)
            task_name = batch['task_name']
         
            warped, flow, orthogonal_loss,classification_loss = model(moving, fixed, task_id, is_training=True)
          
           
            smooth_loss = grad_loss(flow)
            sim_loss = mind_loss(warped, fixed)
    
            loss =  sim_loss +  smooth_loss*0.1 + orthogonal_loss*0.001+classification_loss*0.001

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 30 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            save_path = os.path.join(save_dir, f'epoch_{epoch + 1}.pth')
            torch.save(checkpoint, save_path)
            
           
    

if __name__ == '__main__':
    main()
