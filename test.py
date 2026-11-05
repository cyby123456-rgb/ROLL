python - <<'PY'
import torch
import torchao
import transformers
import compressed_tensors
import sglang

print("torch", torch.__version__)
print("torchao", torchao.__version__)
print("transformers", transformers.__version__)
print("compressed_tensors", compressed_tensors.__version__)
print("sglang", sglang.__version__)

from sglang.srt.entrypoints.engine import Engine
print("SGLang Engine import OK")
PY