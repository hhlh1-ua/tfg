import numpy as np
import torch
import random

global seed


def set_seed(semilla):
    global seed
    seed=semilla
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#     warm_up_generator()


# def warm_up_generator(num_iterations=10):
#     # Generar algunos números aleatorios para "calentar" el generador
#     _ = [random.random() for _ in range(num_iterations)]
#     _ = [np.random.rand() for _ in range(num_iterations)]
#     _ = [torch.rand(1) for _ in range(num_iterations)]



def worker_init_fn(worker_id):
    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

