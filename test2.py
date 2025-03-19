import torch
import time
from tqdm import tqdm

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用的设备: {device}")

# 生成大矩阵
size = 800  # 可以调整大小以控制 GPU 占用率
a = torch.rand(size, size, device=device)
b = torch.rand(size, size, device=device)

# 目标持续时间：30分钟
target_duration = 3600 * 60  # 30分钟（秒）

# 初始化进度条
with tqdm(total=target_duration, desc="trian_time", unit="秒") as pbar:
    start_time = time.time()
    
    # 持续调用矩阵乘法直到达到目标时间
    while time.time() - start_time < target_duration:
        step_start = time.time()
        
        # 执行矩阵乘法
        c = torch.matmul(a, b)
        torch.cuda.synchronize()  # 确保所有 GPU 操作完成
        
        step_end = time.time()
        
        # 更新进度条
        elapsed = step_end - step_start
        pbar.update(elapsed)
        
    print("运行结束，持续时间为 30 分钟")
# import psutil
# import time
# import threading

# def consume_cpu(target_usage=20):
#     interval = 0.1  # Check interval in seconds

#     while True:
#         # Get the CPU utilization
#         cpu_usage = psutil.cpu_percent(interval=interval)
#         sleep_time = (cpu_usage - target_usage) / target_usage * interval
        
#         if sleep_time > 0:
#             time.sleep(sleep_time)
#         else:
#             # Busy wait to consume CPU
#             end_time = time.time() + abs(sleep_time)
#             while time.time() < end_time:
#                 pass

# if __name__ == "__main__":
#     print("Starting CPU stress test to maintain ~20% CPU usage")
    
#     cpu_thread = threading.Thread(target=consume_cpu, args=(20,))
#     cpu_thread.start()
    
#     cpu_thread.join()

#新建的是3，后建的是二 /root/node03-nfs/aaai/hallo/train_audio2_pose/data/filter_split_data.py
# mkdir -p ~/anaconda3
# bash Anaconda3-2024.10-1-Linux-x86_64.sh -b -p ~/anaconda3
