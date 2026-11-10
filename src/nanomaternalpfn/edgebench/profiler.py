"""Observed CPU latency and process RSS; no simulated edge/Android claims."""
import os
import statistics
import time
import torch

def profile(model, x, y_train, repeats=20, warmup=3):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(x,y_train)
        latency=[]
        for _ in range(repeats):
            begin=time.perf_counter()
            model(x,y_train)
            latency.append((time.perf_counter()-begin)*1000)
    result={'runtime':'pytorch','device':str(x.device),'p50_ms':statistics.median(latency),
            'p99_ms':sorted(latency)[min(len(latency)-1,int(.99*len(latency)))],
            'repeats':repeats,'peak_ram_mb':None,'energy_mj':None}
    try:
        import psutil
        result['process_rss_mb']=psutil.Process(os.getpid()).memory_info().rss/1e6
    except ImportError:
        result['process_rss_mb']=None
    return result
