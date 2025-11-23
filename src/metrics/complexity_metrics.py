import os
import time
import torch
from torch import Tensor
from torch import nn
from thop import profile


def compute_model_complexity(model: nn.Module, is_video: bool):
    """
    Количество параметров и MACs
    """
    if is_video:
        input = [torch.rand(size=(1, 32000)), torch.rand((1, 512, 50)), torch.rand((1, 512, 50))]
    else:
        input = [torch.rand(size=(1, 32000))]
    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=(*input,), verbose=False)
    return {
        "коичество параметров": params ,  
        "gmacs": macs / 1e9        
    }
 

def compute_model_size(model_path: str, is_video: bool):
    """
    Размер сохранённого файла модели 
    """
    size_mb = os.path.getsize(model_path) / (1024 ** 2)

    if is_video:
        size_mb += 139 # video model
    return {"размер модели (в МБ)": size_mb}


def compute_memory_usage(model: nn.Module, is_video: bool, device="cuda"):
    """
    Возвращает GPU-память (в GB) для одного forward
    Works ONLY on GPU
    """
    if is_video:
        input = [torch.rand(size=(1, 32000)), torch.rand((1, 512, 50)), torch.rand((1, 512, 50))]
    else:
        input = [torch.rand(size=(1, 32000))]

    model = model.to(device)
    input_sample = [x.to('cuda') for x in input]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        _ = model(*input_sample)
    torch.cuda.synchronize()

    mem_bytes = torch.cuda.max_memory_allocated(device)
    mem_gb = mem_bytes / (1024 ** 3)
    return {"использование памяти (в ГБ)": mem_gb}


def compute_time_per_step(model: nn.Module, device="cuda", n_runs: int = 5, is_video: bool = False):
    """
    Среднее время forward шага (в секундах)
    """
    if is_video:
        input = [torch.rand(size=(1, 32000)), torch.rand((1, 512, 50)), torch.rand((1, 512, 50))]
    else:
        input = [torch.rand(size=(1, 32000))]

    model = model.to(device)
    input_sample = [x.to('cuda') for x in input]

    for _ in range(3):
        _ = model(*input_sample)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(n_runs):
        _ = model(*input_sample)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / n_runs

    return {
        "время на один шаг": elapsed,
        "пропускная способность (в сек)": 1 / elapsed
    }


def summarize_model_performance(model: nn.Module, model_path: str = None, device="cuda", is_video: bool = False):
    """
    Все характеристики модели:
    - количество параметров
    - GMACs
    - размер модели
    - использование памяти
    - время на один шаг
    - пропускная способность
    """
    results = {}
    results.update(compute_model_complexity(model, is_video))
    if model_path is not None:
        results.update(compute_model_size(model_path, is_video))
    if torch.cuda.is_available():
        results.update(compute_memory_usage(model, is_video, device))
        results.update(compute_time_per_step(model, device, is_video))
    else:
        results.update(compute_time_per_step(model, device="cpu", is_video=is_video))
    return results