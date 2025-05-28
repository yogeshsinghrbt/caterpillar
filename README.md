# Caterpillar

### Accelerating low bit models on consumer grade GPUs 

- Reduces memory requirements by 3x compared to FP16
- Includes efficient 3 bit split K matmul kernel for on the fly dequantization



## Usage

3 Bit Quantization

```
import torch 
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from quantize.pre_quant import run_awq
from quantize.quantizer import real_quantize_model_weight

model_path = "facebook/opt-1.3b"

q_config = {
    "zero_point": True,
    "q_group_size": 128,
}

# Init model on CPU:
kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

enc = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(model_path, config=config, trust_remote_code=True, **kwargs)
model.eval()

run_awq(model, enc, w_bit=3, q_config=q_config, n_samples=128, seqlen=512)

real_quantize_model_weight(model, w_bit=3, q_config=q_config)

```

## Benchmark

These benchmarks showcase the performance and memory efficiency during token generation (decoding) phase.

Performance can vary depending on your hardware—not just across GPUs, but CPUs as well. 

Model : opt 1.3b 

Hardware: RTX 4060 Mobile

| Quantization  | Batch Size | Memory (MB) | token/sec |
|---------------|------------|-------------|-----------| 
|  fp16         |1           | 2539        |  41.2    
|  fp16         | 10         | 2727        |  107
| 4 Bit         | 1          | 874         |  23.83
| 4 Bit         | 10          | 1036         |  120.19
| 4 Bit, Dequantize on the fly  | 1          | 848         |  23
| 4 Bit, Dequantize on the fly  | 10          | 1036         |  37
| 3 Bit         | 1          | 778        |  25
| 3 Bit         | 10          | 939         |  109
| 3 Bit, Dequantize on the fly  | 1          |  755      |  16.35
| 3 Bit, Dequantize on the fly  | 10          | 939         |  23.60


## Reference

#### AWQ

@inproceedings{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Chen, Wei-Ming and Wang, Wei-Chen and Xiao, Guangxuan and Dang, Xingyu and Gan, Chuang and Han, Song},
  booktitle={MLSys},
  year={2024}
}
