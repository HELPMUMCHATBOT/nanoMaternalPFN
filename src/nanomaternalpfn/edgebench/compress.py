"""Dynamic quantization baseline; not measured mobile INT8 KV caching."""
import torch

def dynamic_int8(model):
    return torch.ao.quantization.quantize_dynamic(model.cpu().eval(),{torch.nn.Linear},dtype=torch.qint8)
