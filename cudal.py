import os, sys, torch
print("Python:", sys.executable)
print("torch.__version__:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)   # None => CPU-only build
print("cudnn:", torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None)
print("device_count:", torch.cuda.device_count())
print("is_available:", torch.cuda.is_available())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))