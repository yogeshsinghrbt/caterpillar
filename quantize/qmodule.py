import math
import torch
import torch.nn as nn
from torch.autograd import Function
#import awq_inference_engine  # with CUDA kernels

from .triton.quant_gemm_3bit import awq_quant_gemm_triton_3bit, awq_dequantize_triton_3bit
from .triton.quant_gemm_4bit import awq_quant_gemm_triton_4bit, awq_dequantize_triton_4bit

# Adapted from https://github.com/compressa-ai/AutoAWQ/tree/dev
class WQLinearMMFunction(Function):
    @staticmethod
    # ctx is the first argument to forward
    def forward(
        ctx,
        x,
        qweight,
        qzeros,
        scales,
        w_bit=4,
        group_size=128,
        bias=None,
        out_features=0,
    ):
        out_shape = x.shape[:-1] + (out_features,)
        x = x.to(torch.float16)

        # Use when input dimension is low
        # out = awq_dequantize_triton_4bit(qweight, scales, qzeros)
        # out = torch.matmul(x, out.to(x.dtype))


        if w_bit == 4:
            out = awq_quant_gemm_triton_4bit(x.reshape(-1, x.shape[-1]), qweight, scales, qzeros, split_k_iters=8)
        elif w_bit == 3:
            out = awq_quant_gemm_triton_3bit(x.reshape(-1, x.shape[-1]), qweight, scales, qzeros, split_k_iters=8)
        

        out = out + bias if bias is not None else out
        out = out.reshape(out_shape)

        # always want 3D tensor if tensor is 2D
        if len(out.shape) == 2:
            out = out.unsqueeze(0)

        return out



class WQLinear_GEMM(nn.Module):
    def __init__(
        self, w_bit, group_size, in_features, out_features, bias, dev
    ):
        super().__init__()

        if w_bit not in [3, 4]:
            raise NotImplementedError("Only 3-bit are supported for now.")

        #print ("WQLinear_GEMM")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.padding = 0


        pack_num = 32 // self.w_bit

        # Calculate padding needed to make out_features a multiple of pack_num
        self.original_out_features = out_features
        self.padded_out_features = out_features

        if out_features % pack_num != 0:
            self.padding = pack_num - (out_features % pack_num)
            self.padded_out_features = out_features + self.padding
        else:
            self.padded_out_features = out_features

        assert self.padded_out_features % (32 // self.w_bit) == 0
            

        self.register_buffer(
            "qweight",
            torch.zeros(
                (in_features, self.padded_out_features // pack_num),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (in_features // self.group_size, self.padded_out_features // pack_num),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (in_features // self.group_size, self.padded_out_features),
                dtype=torch.float16,
                device=dev,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (self.padded_out_features),
                    dtype=torch.float16,
                    device=dev,
                ),
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None
    ):
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd
            return awq_linear

        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None
        scale_zeros = zeros * scales

        pack_num = 32 // awq_linear.w_bit

        
        #-------------------------------------------- intweight --------------------------------------------------------

        intweight = []
        
        for idx in range(awq_linear.in_features):
            intweight.append(
                torch.round(
                    (linear.weight.data[:, idx] + scale_zeros[idx // group_size])
                    / scales[idx // group_size]
                ).to(torch.int)[:, None]
            )
            
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.to(dtype=torch.int32)

        #torch.save(intweight, "intweight.pt")


        # If we need padding, pad the quantized weights
        if awq_linear.padded_out_features > linear.out_features:
            padding_size = awq_linear.padded_out_features - linear.out_features
            padding = torch.zeros(
                (intweight.shape[0], padding_size),
                dtype=intweight.dtype,
                device=intweight.device
            )
            intweight = torch.cat([intweight, padding], dim=1)

        #--------------------------------------------------- qweight  -------------------------------------------------------------

        qweight = torch.zeros(
            (intweight.shape[0], intweight.shape[1] // pack_num),
            dtype=torch.int32,
            device=linear.weight.device,
        )

        #print(intweight)
        for col in range(intweight.shape[1] // pack_num):
            
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            elif awq_linear.w_bit == 3:
                order_map = [0, 2, 4, 6,8, 1, 3, 5, 7, 9]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
                
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear.w_bit)
                
        awq_linear.qweight = qweight

        #------------------------------------------------- qzeros ----------------------------------------------------------------

        zeros = zeros.to(dtype=torch.int32)#, device=best_device)

        # If we need padding, pad the quantized weights
        if awq_linear.padded_out_features > linear.out_features:
            padding_size = awq_linear.padded_out_features - linear.out_features
            padding = torch.zeros(
                (zeros.shape[0], padding_size),
                dtype=intweight.dtype,
                device=intweight.device
            )
            zeros = torch.cat([zeros, padding], dim=1)


        #torch.save(zeros, "zeros_unpacked.pt")
        
        qzeros = torch.zeros(
            (zeros.shape[0], zeros.shape[1] // pack_num),
            dtype=torch.int32,
            device=zeros.device,
        )

        for col in range(zeros.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            elif awq_linear.w_bit == 3:
                order_map = [0, 2, 4, 6,8, 1, 3, 5, 7, 9]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qzero_col = zeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.w_bit)
        awq_linear.qzeros = qzeros
        

        #------------------------------------------ scales and bias ---------------------------------------------------

        # If we need padding, pad the quantized weights
        if awq_linear.padded_out_features > linear.out_features:
            padding_size = awq_linear.padded_out_features - linear.out_features
            padding = torch.zeros(
                (scales.shape[0], padding_size),
                dtype=intweight.dtype,
                device=intweight.device
            )
            scales = torch.cat([scales, padding], dim=1)

        
        awq_linear.scales = scales.clone().half().contiguous()
        #awq_linear.scales = awq_linear.scales.t().contiguous()
        
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()


        #--------------------------------------------------------------------------------------------------
        

        # print ("awq_linear.qweight", awq_linear.qweight.shape, awq_linear.qweight.device)
        # print ("awq_linear.scales", awq_linear.scales.shape, awq_linear.scales.device)
        # print ("awq_linear.qzeros", awq_linear.qzeros.shape, awq_linear.qzeros.device)
        
        

        return awq_linear

    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features,)

        input_dtype = x.dtype
        if input_dtype != torch.float16:
            x = x.half()

        with torch.no_grad():
            out = WQLinearMMFunction.apply(
                x,
                self.qweight,
                self.qzeros,
                self.scales,
                self.w_bit,
                self.group_size,
                self.bias,
                self.out_features
            )

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)

        return out.reshape(out_shape)

    def extra_repr(self) -> str:
        return (
            "in_features={}, out_features={}, bias={}, w_bit={}, group_size={}".format(
                self.in_features,
                self.out_features,
                self.bias is not None,
                self.w_bit,
                self.group_size,
            )
        )