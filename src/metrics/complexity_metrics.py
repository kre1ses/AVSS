import os
import time
import torch
from torch import Tensor
from torch import nn
from thop import profile


def compute_model_complexity(model: nn.Module, input_sample: torch.Tensor):
    """
    Количество параметров и MACs
    """
    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=(input_sample,), verbose=False)
    return {
        "params_m": params / 1e6,   
        "gmacs": macs / 1e9        
    }


def compute_model_size(model_path: str):
    """
    Размер сохранённого файла модели 
    """
    size_mb = os.path.getsize(model_path) / (1024 ** 2)
    return {"model_size_mb": size_mb}


def compute_memory_usage(model: nn.Module, input_sample: Tensor, device="cuda"):
    """
    Возвращает GPU-память (в GB) для одного forward
    """
    model = model.to(device)
    input_sample = input_sample.to(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        _ = model(input_sample)
    torch.cuda.synchronize()

    mem_bytes = torch.cuda.max_memory_allocated(device)
    mem_gb = mem_bytes / (1024 ** 3)
    return {"memory_gb": mem_gb}


def compute_time_per_step(model: nn.Module, input_sample: torch.Tensor, device="cuda", n_runs: int = 5):
    """
    Среднее время forward шага (в секундах)
    """
    model = model.to(device)
    input_sample = input_sample.to(device)

    for _ in range(3):
        _ = model(input_sample)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(n_runs):
        _ = model(input_sample)
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / n_runs

    return {
        "time_per_step_s": elapsed,
        "throughput_samples_per_s": input_sample.size(0) / elapsed
    }


def summarize_model_performance(model: nn.Module, input_sample: torch.Tensor, model_path: str = None, device="cuda"):
    """
    Все характеристики модели:
    - количество параметров
    - MACs
    - размер модели
    - использование памяти
    - время на один шаг
    - пропускная способность
    """
    results = {}
    results.update(compute_model_complexity(model, input_sample))
    if model_path is not None:
        results.update(compute_model_size(model_path))
    if torch.cuda.is_available():
        results.update(compute_memory_usage(model, input_sample, device))
        results.update(compute_time_per_step(model, input_sample, device))
    else:
        results.update(compute_time_per_step(model, input_sample, device="cpu"))
    return results