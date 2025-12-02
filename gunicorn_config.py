import multiprocessing
import os

bind = "0.0.0.0:7860"

cores = multiprocessing.cpu_count()
default_workers = (cores * 2) + 1

max_workers = int(os.getenv("MAX_WORKERS", "4"))

workers = min(default_workers, max_workers)

worker_class = "uvicorn.workers.UvicornWorker"
timeout = 120
keepalive = 5

loglevel = "info"
accesslog = "-"
errorlog = "-"