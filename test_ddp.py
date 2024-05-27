# see pytorch model.state_dict() and model.module.state_dict()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import torch
import torch.nn as nn
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from loguru import logger
from config_ini import get_cfg_defaults
from tqdm import tqdm

class MyDataset(Dataset):
    def __init__(self, num_samples=400, size=3, device=0):
        self.device = device
        self.data = torch.randn(num_samples, 3, size, size)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx].to(self.device)

class Net(nn.Module):
    def __init__(self, size=3):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(3, 2, 1)
        self.mlp = nn.Linear(size, 1)
        self.fc = nn.Linear(1, 1)
        
    def forward(self, x):
        # device = self.conv.weight.device
        # x = x.to(device)
        x = self.conv(x)
        x = x.reshape(-1, x.shape[1] * x.shape[2], x.shape[3])
        x = self.mlp(x)
        x = self.fc(x)
        return x
    
class Trainer:
    def __init__(self, rank, args, world_size):
        self.parse_args(args)
        self.init_distributed(rank, world_size)
        self.init_datasets()
        self.init_model()
        # self.init_writer()
        self.train()
        # self.cleanup()

    def parse_args(self, args):
        self.args = args

    def init_distributed(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        logger.info('Initializing distributed')
        port = int(self.args.train.distributed_port)
        #os.environ['MASTER_ADDR'] = self.args.train.distributed_addr 
        while 1:
            try:
                #os.environ['MASTER_PORT'] = str(port)
                dist_url = f'tcp://{self.args.train.distributed_addr}:{port}'
                dist.init_process_group("nccl", init_method=dist_url,rank=rank, world_size=world_size)
                print(f'Using Port {port}')
                break
            except Exception as e:
                print(e)
                print(f'Port {port} Used')
                port += 1

    def init_datasets(self):
        logger.info('Initializing datasets')

        self.dataset = MyDataset(device='cpu')
        # Matting dataloaders:
        self.datasampler = DistributedSampler(
            dataset=self.dataset,
            rank=self.rank,
            num_replicas=self.world_size,
            shuffle=True)
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=2,
            num_workers=4,
            sampler=self.datasampler,
            pin_memory=True)

    def init_model(self):
        logger.info('Initializing model')
        self.model = Net().to(self.rank)
        # self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = DDP(self.model, device_ids=[self.rank])
        if self.rank == 0:
            # print(self.model)
            print(self.model.module)
    
    def train(self):
        for epoch in range(10):
            if self.rank == 0:
                logger.info(f'Epoch {epoch} Starting')
            for i, batch in enumerate(tqdm(self.dataloader)):
                #if self.rank == 0:
                    #print(self.rank, i, batch)
                    # @print(self.rank)
                batch = batch.to(self.rank)
                self.model(batch)

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    args = get_cfg_defaults()
    mp.spawn(
        fn=Trainer,
        args=(args, world_size,),
        nprocs=world_size,
        join=True)
