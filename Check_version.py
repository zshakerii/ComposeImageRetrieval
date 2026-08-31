import sys
import torch
import PIL
import numpy as np
import transformers
try:
    import clip
except ImportError:
    clip = None
try:
    import open_clip
except ImportError:
    open_clip = None

def get_version(package, name):
    try دقیق آن‌ها را چاپ می‌کند.

### راهنما:
۱. یک فایل جدید با نام `check_versions.py` بسازید.
۲. کد زیر را در آن کپی کنید.
۳. در ترمینال یا CMD، دستور `python check_versions.py` را اجرا کنید.
```python
import sys
import torch
import PIL
import numpy as np
import transformers
try:
import clip
except ImportError:
clip = None
try:
import open_clip
except ImportError:
open_clip = None

def get_version(package, name):
try:
return package.__version__
except AttributeError:
return "Not found or no version attribute"

print(f"--- System Versions ---")
print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")

# بررسی CUDA
if torch.cuda.is_available():
print(f"CUDA: {torch.version.cuda} (Device: {torch.cuda.get_device_name(0)})")
else:
print("CUDA: Not available / Not found")

print(f"NumPy: {get_version(np, 'NumPy')}")
print(f"Pillow: {get_version(PIL, 'Pillow')}")
print(f"Transformers: {get_version(transformers, 'Transformers')}")

# بررسی CLIP و OpenCLIP
if clip:
# گاهی اوقات clip ممکن است version نداشته باشد
print(f"OpenAI CLIP: Installed (Check manual install location)")
else:
print("OpenAI CLIP: Not installed")

if open_clip:
print(f"OpenCLIP: {get_version(open_clip, 'OpenCLIP')}")
else:
print("OpenCLIP: Not installed")
