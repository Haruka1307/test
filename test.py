import torch
import torch.nn as nn
import time

def test_8gpu_performance():
    # 检查GPU数量
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 8, f"需要至少8张GPU，当前只有 {num_gpus} 张"
    print(f"✅ 检测到 {num_gpus} 张GPU: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")

    # 设置每张GPU的测试数据（显存占用约10GB）
    batch_size = 1024
    dim = 8192  # 矩阵大小 (dim x dim)
    dtype = torch.float32

    # 在每张GPU上创建随机矩阵
    inputs = [torch.randn(batch_size, dim, dim, dtype=dtype).to(f'cuda:{i}') for i in range(num_gpus)]
    
    # 定义一个简单的模型（矩阵乘法）
    model = nn.Linear(dim, dim).cuda()
    model = nn.DataParallel(model, device_ids=range(num_gpus))  # 数据并行

    # 预热GPU（避免首次运行速度偏差）
    for _ in range(3):
        _ = model(inputs[0])

    # 正式测试
    start_time = time.time()
    for i in range(10):  # 运行10次迭代
        outputs = model(inputs[i % num_gpus])  # 轮询使用不同GPU的数据
    elapsed_time = time.time() - start_time

    # 输出结果
    print(f"\n⏱️  总计算时间: {elapsed_time:.2f}秒")
    print(f"🚀 平均每次迭代时间: {elapsed_time / 10:.4f}秒")
    print("🎯 测试完成！运行 `nvidia-smi` 查看显存占用。")

if __name__ == "__main__":
    test_8gpu_performance()