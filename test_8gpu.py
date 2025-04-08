import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import argparse

# 示例模型定义
class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(128 * 32 * 32, 10)  # 假设输入是32x32图像
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 示例数据集
class TestDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.data = torch.randn(num_samples, 3, 32, 32)
        self.labels = torch.randint(0, 10, (num_samples,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()

def prepare_dataloader(dataset, batch_size, rank, world_size):
    """准备分布式数据加载器"""
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=sampler
    )
    return dataloader

def test_model(rank, world_size, args):
    """测试模型的主函数"""
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    
    # 创建模型并移动到当前GPU
    model = TestModel().to(rank)
    
    # 使用DDP包装模型
    ddp_model = DDP(model, device_ids=[rank])
    
    # 准备数据集和数据加载器
    dataset = TestDataset(num_samples=1000)
    dataloader = prepare_dataloader(dataset, args.batch_size, rank, world_size)
    
    # 测试循环
    ddp_model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(rank), labels.to(rank)
            outputs = ddp_model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # 收集所有进程的结果
    total_tensor = torch.tensor(total).to(rank)
    correct_tensor = torch.tensor(correct).to(rank)
    
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        accuracy = 100 * correct_tensor.item() / total_tensor.item()
        print(f'Accuracy of the model on test images: {accuracy:.2f}%')
    
    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for testing')
    args = parser.parse_args()
    
    world_size = 8  # 使用8个GPU
    mp.spawn(
        test_model,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()