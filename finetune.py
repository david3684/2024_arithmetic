import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.datasets as datasets
from torchvision.datasets import DTD as TorchvisionDTD
import open_clip
from torch.utils.data import DataLoader
from transformers import AdamW, get_cosine_schedule_with_warmup
from tqdm import tqdm

class DTD:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('/data2/arithmetic/data'),
                 batch_size=32,
                 num_workers=16,
                 partition=1,
                 distributed=False,
                 rank=0):
        # Data loading code
        self.train_dataset = TorchvisionDTD(
            root=location, split='train', partition=partition, transform=preprocess, download=True)
        
        if distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, rank=rank)
        else:
            self.train_sampler = None

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            shuffle=(self.train_sampler is None),
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.test_dataset = TorchvisionDTD(
            root=location, split='val', partition=partition, transform=preprocess, download=True)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        idx_to_class = dict((v, k)
                            for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace(
            '_', ' ') for i in range(len(idx_to_class))]

def setup(rank, world_size):
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main_worker(rank, world_size):
    setup(rank, world_size)
    
    device = torch.device(f"cuda:{rank}")

    model_name = "ViT-L-14"
    pretrained_model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="laion2b_s32b_b82k")
    for name, param in pretrained_model.named_parameters():
        if 'classifier' in name:  
            param.requires_grad = False

    pretrained_model = pretrained_model.to(device)
    
    
    pretrained_model = DDP(pretrained_model, device_ids=[rank], find_unused_parameters=True)

    dtd_data = DTD(preprocess, batch_size=32, num_workers=16, distributed=True, rank=rank)

    learning_rate = 1e-5
    num_iterations = 2000
    warmup_steps = 200
    total_steps = num_iterations
    weight_decay = 0.01

    optimizer = AdamW(pretrained_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    criterion = torch.nn.CrossEntropyLoss()

    pretrained_model.train()

    global_step = 0
    total_epochs = num_iterations // len(dtd_data.train_loader)
    
    for epoch in range(total_epochs):
        if dtd_data.train_sampler:
            dtd_data.train_sampler.set_epoch(epoch)
        
        progress_bar = tqdm(enumerate(dtd_data.train_loader), total=len(dtd_data.train_loader), desc=f"Epoch {epoch+1}/{total_epochs}", position=rank)

        for batch_idx, (images, labels) in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = pretrained_model(images)
            logits = outputs[0]  
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            global_step += 1

            if rank == 0:
                progress_bar.set_postfix({"loss": loss.item(), "step": global_step})

            if global_step >= total_steps:
                break
        if global_step >= total_steps:
            break

    if rank == 0:
        torch.save(pretrained_model.state_dict(), "finetuned_clip_vit_l14_dtd.pth")

    cleanup()

def run_ddp(world_size):
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    world_size = 1
    run_ddp(world_size)
