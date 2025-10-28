# from torch._higher_order_ops.associative_scan import associative_scan
# import torch
# import torch.nn.functional as F


# # apply parallel scan in the second dimension
# class AScan(torch.autograd.Function): # (b c l p)
#     @staticmethod
#     def scan_op(i, j):
#         g_i, x_i = i
#         g_j, x_j = j
#         return g_j * g_i, g_j * x_i + x_j

#     @torch.compile
#     @staticmethod
#     @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
#     def forward(ctx, g, x):
#         @torch._dynamo.disable
#         def _assoc_scan(g, x):
#             return associative_scan(AScan.scan_op, (g, x), dim=2)
#         _, x_scan = _assoc_scan(g, x)
#         #_, x_scan = associative_scan(AScan.scan_op, (g, x), dim=2)
#         # import torch._higher_order_ops.associative_scan as ascan_module
#         # with torch._dynamo.config.patch(suppress_errors=True):
#         #     _, x_scan = associative_scan(AScan.scan_op, (g, x), dim=2)
#         ctx.save_for_backward(g, x_scan)
#         return x_scan

#     @torch.compile
#     @staticmethod
#     @torch.amp.custom_bwd(device_type="cuda") # (b c l p)
#     def backward(ctx, grad):
#         g, x_scan = ctx.saved_tensors
#         g = F.pad(g, (0, 0, -1, 1))
#         @torch._dynamo.disable
#         def _assoc_scan(g, x):
#             return associative_scan(AScan.scan_op, (g, x), dim=2)
#         _, x_grad = _assoc_scan(g, grad)
#         #  _, x_grad = associative_scan(AScan.scan_op, (g, grad), dim=2, reverse=True)
#         g_grad = torch.zeros_like(x_scan)
#         g_grad[:, :, 1:].add_(x_scan[:, :, :-1] * x_grad[:, :, 1:])
#         return g_grad, x_grad

# ascan = AScan.apply
# import torch

# def ascan_seq(g, x):
#     """
#     Sequential prefix-scan (autograd-friendly).
#     g, x: tensors with shape (b, c, L, p) (or generally (..., L, p))
#     scan along dim=2 (index 2).
#     Returns prefix_x same shape as x.
#     """
#     if g.dim() < 3:
#         raise RuntimeError("ascan_seq: expected g with dim >= 3 (b,c,L,...)")

#     L = g.size(2)
#     prefix_g = torch.empty_like(g)
#     prefix_x = torch.empty_like(x)

#     prefix_g[..., 0, :] = g[..., 0, :].clone()
#     prefix_x[..., 0, :] = x[..., 0, :].clone()

#     for t in range(1, L):
#         prev_g = prefix_g[..., t - 1, :]
#         prev_x = prefix_x[..., t - 1, :]
#         cur_g = g[..., t, :]
#         cur_x = x[..., t, :]
#         new_g = cur_g * prev_g
#         new_x = cur_g * prev_x + cur_x
#         prefix_g[..., t, :] = new_g
#         prefix_x[..., t, :] = new_x

#     return prefix_x

# # alias to match old name
# ascan = ascan_seq
import torch

def ascan_seq(g, x):
    if g.dim() < 3:
        raise RuntimeError("ascan_seq: expected g with dim >= 3 (b,c,L,...)")

    L = g.size(2)
    prefix_g = torch.empty_like(g)
    prefix_x = torch.empty_like(x)

    prefix_g[..., 0, :] = g[..., 0, :]
    prefix_x[..., 0, :] = x[..., 0, :]

    for t in range(1, L):
        prev_g = prefix_g[..., t - 1, :]
        prev_x = prefix_x[..., t - 1, :]
        cur_g = g[..., t, :]
        cur_x = x[..., t, :]
        new_g = cur_g * prev_g
        new_x = cur_g * prev_x + cur_x
        # 关键：禁止这些写操作进入 autograd 路径
        with torch.no_grad():
            prefix_g[..., t, :] = new_g
            prefix_x[..., t, :] = new_x

    return prefix_x
ascan = ascan_seq
