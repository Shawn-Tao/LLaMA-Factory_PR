from transformers import AutoModelForVision2Seq
import torch.nn as nn

model = AutoModelForVision2Seq.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct",
    trust_remote_code=True
)

names = set()
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        names.add(name.split(".")[-1])

for n in sorted(names):
    print(n)