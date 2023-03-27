from pynvml import (
    nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
)
from numba import cuda
import torch


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")



def free_gpu_cache(hard=False, device_id=None):

    print("Initial GPU usage: ", print_gpu_utilization())
    gc.collect()
    torch.cuda.empty_cache()
    if hard:
        cuda.select_device(device_id)
        cuda.close()
        cuda.select_device(0)
    print("GPU usage post cleaning", print_gpu_utilization())

