import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    # def fasterquant(
    #     self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    # ):
    #     W = self.layer.weight.data.clone()
    #     if isinstance(self.layer, nn.Conv2d):
    #         W = W.flatten(1)
    #     if isinstance(self.layer, transformers.Conv1D):
    #         W = W.t()
    #     W = W.float()

    #     tick = time.time()

    #     if not self.quantizer.ready():
    #         self.quantizer.find_params(W, weight=True)

    #     H = self.H
    #     del self.H
    #     dead = torch.diag(H) == 0
    #     H[dead, dead] = 1
    #     W[:, dead] = 0

    #     if static_groups:
    #         import copy
    #         groups = []
    #         for i in range(0, self.columns, groupsize):
    #             quantizer = copy.deepcopy(self.quantizer)
    #             quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
    #             groups.append(quantizer)

    #     if actorder:
    #         perm = torch.argsort(torch.diag(H), descending=True)
    #         W = W[:, perm]
    #         H = H[perm][:, perm]
    #         invperm = torch.argsort(perm)

    #     Losses = torch.zeros_like(W)
    #     Q = torch.zeros_like(W)

    #     damp = percdamp * torch.mean(torch.diag(H))
    #     diag = torch.arange(self.columns, device=self.dev)
    #     H[diag, diag] += damp
    #     H = torch.linalg.cholesky(H)
    #     H = torch.cholesky_inverse(H)
    #     H = torch.linalg.cholesky(H, upper=True)
    #     Hinv = H

    #     for i1 in range(0, self.columns, blocksize):
    #         i2 = min(i1 + blocksize, self.columns)
    #         count = i2 - i1

    #         W1 = W[:, i1:i2].clone()
    #         Q1 = torch.zeros_like(W1)
    #         Err1 = torch.zeros_like(W1)
    #         Losses1 = torch.zeros_like(W1)
    #         Hinv1 = Hinv[i1:i2, i1:i2]

    #         for i in range(count):
    #             w = W1[:, i]
    #             d = Hinv1[i, i]

    #             if groupsize != -1:
    #                 if not static_groups:
    #                     if (i1 + i) % groupsize == 0:
    #                         self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
    #                 else:
    #                     idx = i1 + i
    #                     if actorder:
    #                         idx = perm[idx]
    #                     self.quantizer = groups[idx // groupsize]

    #             q = quantize(
    #                 w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
    #             ).flatten()
    #             Q1[:, i] = q
    #             Losses1[:, i] = (w - q) ** 2 / d ** 2

    #             err1 = (w - q) / d
    #             W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
    #             Err1[:, i] = err1

    #         Q[:, i1:i2] = Q1
    #         Losses[:, i1:i2] = Losses1 / 2

    #         W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    #         if DEBUG:
    #             self.layer.weight.data[:, :i2] = Q[:, :i2]
    #             self.layer.weight.data[:, i2:] = W[:, i2:]
    #             print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
    #             print(torch.sum(Losses))

    #     torch.cuda.synchronize()
    #     print('time %.2f' % (time.time() - tick))
    #     print('error', torch.sum(Losses).item())

    #     if actorder:
    #         Q = Q[:, invperm]

    #     if isinstance(self.layer, transformers.Conv1D):
    #         Q = Q.t()
    #     self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
    #     if DEBUG:
    #         print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        rows, cols = W.shape
        print('Quantizing %d x %d matrix' % (rows, cols))

        # 逻辑：将权重 reshape 为 [Rows, Cols/4, 4]，在最后一个维度取 Top-2

        if cols % 4 == 0:
            W_view = W.view(rows, -1, 4)
            # 打印W_view_sparse的前10行
            print(W_view[0, 0])

            magnitude = W_view.abs()
            
            _, indices = torch.topk(magnitude, k=2, dim=2)

            mask_view = torch.zeros_like(W_view, dtype=torch.bool)
            mask_view.scatter_(2, indices, True)
            print(mask_view[0, 0])

            mask = mask_view.view(rows, cols)

            W = W * mask.float()
        else:
            print(f"Warning: Layer columns {cols} not divisible by 4, skipping sparsity.")

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    # def fasterquant(
    #     self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    # ):
    #     """
    #     SparseGPT 实现: 2:4 稀疏化 + Hessian 误差补偿
    #     PPL 预期: 恢复到 16-18 左右
    #     """
    #     # 1. 准备数据
    #     W = self.layer.weight.data.clone()
    #     if isinstance(self.layer, nn.Conv2d):
    #         W = W.flatten(1)
    #     if isinstance(self.layer, transformers.Conv1D):
    #         W = W.t()
    #     W = W.float()

    #     tick = time.time()
    #     rows, cols = W.shape
        
    #     # [关键] 恢复 Hessian 矩阵相关逻辑
    #     H = self.H
    #     del self.H
    #     dead = torch.diag(H) == 0
    #     H[dead, dead] = 1
    #     W[:, dead] = 0

    #     # 计算 Hessian 的逆矩阵 (用于指导如何修复误差)
    #     damp = percdamp * torch.mean(torch.diag(H))
    #     diag = torch.arange(self.columns, device=self.dev)
    #     H[diag, diag] += damp
    #     H = torch.linalg.cholesky(H)
    #     H = torch.cholesky_inverse(H)
    #     H = torch.linalg.cholesky(H, upper=True)
    #     Hinv = H

    #     mask_cache = None

    #     # --- Block-wise 循环 ---
    #     for i1 in range(0, self.columns, blocksize):
    #         i2 = min(i1 + blocksize, self.columns)
    #         count = i2 - i1

    #         W1 = W[:, i1:i2].clone()
    #         Q1 = torch.zeros_like(W1)
    #         Err1 = torch.zeros_like(W1)
    #         Hinv1 = Hinv[i1:i2, i1:i2]

    #         # --- Column-wise 循环 (逐列处理) ---
    #         for i in range(count):
    #             w = W1[:, i]
    #             d = Hinv1[i, i]

    #             # 1. 生成 2:4 稀疏掩码 (每 4 列一次)
    #             if i % 4 == 0:
    #                 if i + 4 <= count:
    #                     w_group = W1[:, i:i+4]
    #                     # 找到幅度最大的 2 个
    #                     _, indices = torch.topk(w_group.abs(), k=2, dim=1)
    #                     mask_cache = torch.zeros_like(w_group, dtype=torch.bool)
    #                     mask_cache.scatter_(1, indices, True)
    #                 else:
    #                     mask_cache = torch.ones((rows, count - i), dtype=torch.bool, device=self.dev)
                
    #             # 取出当前列的 mask
    #             mask_col = mask_cache[:, i % 4].float()

    #             # 2. 执行稀疏化 (这里相当于 Quantize 操作)
    #             # q 就是被剪枝后的权重 (保留的值不变，剪掉的值变0)
    #             q = w * mask_col

    #             # 3. [核心] 计算误差并补偿
    #             # err 是被剪掉的权重值 (w - 0 = w)
    #             Q1[:, i] = q
    #             err1 = (w - q) / d 
                
    #             # 将误差扩散给后面还没处理的权重
    #             # 这一步是把 PPL 从 413 救回 16 的关键
    #             W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
    #             Err1[:, i] = err1

    #         # 更新全局权重
    #         W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    #     # 收尾
    #     print('time %.2f' % (time.time() - tick))

    #     if isinstance(self.layer, transformers.Conv1D):
    #         W = W.t()
    #     self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
