import torch
import torch.nn as nn
from tqdm import tqdm
import gc

import torch.nn.functional as F

from utils.module import set_op_by_name

EMBEDDING_KEYWORDS = ["embed"]
LM_HEAD_KEYWORDS = ["lm_head", "embed_out", "output"]

# core quantization method (simulated quantization)
def pseudo_quantize_tensor(
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w


# use sliding window for quantization
def pseudo_quantize_tensor_2d(
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    q_group_size=(1, 128)
    org_w_shape = w.shape

    window_size = q_group_size
    stride = q_group_size
    
    windows = w.unfold(0, window_size[0], stride[0]).unfold(1, window_size[1], stride[1])
    num_windows = windows.shape[0] * windows.shape[1]
    windows_reshaped = windows.contiguous().view(num_windows, *window_size)

    window_max = windows_reshaped.view(num_windows, -1).max(dim=1).values
    window_min = windows_reshaped.view(num_windows, -1).min(dim=1).values

    max_int = 2**n_bit - 1
    min_int = 0

    scales = (window_max - window_min).clamp(min=1e-5) / max_int
    zeros = (-torch.round(window_min / scales)).clamp_(min_int, max_int)
    
    assert torch.isnan(scales).sum() == 0

    padded_w = (torch.clamp(torch.round(windows_reshaped / scales[:, None, None]) + zeros[:, None, None], min_int, max_int) - zeros[:, None, None]) * scales[:, None, None]

    afterfold = padded_w.reshape([windows.shape[0], windows.shape[1], *window_size])
    afterfold_permuted = afterfold.permute(0, 2, 1, 3)

    afterfold_permuted = afterfold_permuted.reshape(
        windows.shape[0]*windows.shape[2],
        windows.shape[1]*windows.shape[3]
    )
    
    assert torch.isnan(w).sum() == 0

    
    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w

@torch.no_grad()
def pseudo_quantize_model_weight(
    model,
    w_bit,
    q_config,
):
    from .pre_quant import get_blocks, get_named_linears

    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = get_named_linears(layers[i])
        for n, m in named_linears.items():
            m.cuda()
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, **q_config)
            m.cpu()

            

@torch.no_grad()
def real_quantize_model_weight(model, w_bit, q_config, init_only=False):
    from .qmodule import WQLinear_GEMM
    from .pre_quant import get_blocks, get_named_linears

    assert q_config["zero_point"], "We only support zero_point quantization now."

    layers = get_blocks(model)
    for i in tqdm(
        range(len(layers)),
        desc="real weight quantization..." + ("(init only)" if init_only else ""),
    ):
        layer = layers[i]
        named_linears = get_named_linears(layer)

        for name, module in named_linears.items():
            #print (name, module)
            if init_only:
                q_linear = WQLinear.from_linear(
                    module, w_bit, q_config["q_group_size"], True
                )
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
            else:
                module.cuda()
                
                module.weight.data, scales, zeros = pseudo_quantize_tensor(
                    module.weight.data, n_bit=w_bit, get_scale_zp=True, **q_config
                )
                
                scales = scales.t().contiguous()
                zeros = zeros.t().contiguous()
                
                q_linear = WQLinear_GEMM.from_linear(
                    module, w_bit, q_config["q_group_size"], False, scales, zeros
                )

                module.cpu()
                q_linear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, q_linear)
                torch.cuda.empty_cache()
                gc.collect()

    torch.cuda.empty_cache()
    gc.collect()
