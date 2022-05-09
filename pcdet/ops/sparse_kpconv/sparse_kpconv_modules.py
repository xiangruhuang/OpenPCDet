import torch
from .sparse_kpconv_cuda import (
    sparse_kpconv_gpu,
    tensor_outer_gpu
)

from torch.autograd import Function, Variable

class SparseKPConv(Function):

    @staticmethod
    def forward(ctx, X, W, a):
        """
        Args:
            X [N, D_in]
            W [K, D_out, D_in]
            a [N, K]

        Returns:
            Y [N, D_out]
        """
        Y = X.new_zeros(X.shape[0], W.shape[1])
        sparse_kpconv_gpu(X, W, a, Y)
        ctx.for_backward = (X, W, a) #ctx.save_for_backward(X, W, a)
        return Y

    @staticmethod
    def backward(ctx, grad_Y):
        """
        Args:
            grad_Y [N, D_out]
        
        Returns:
            grad_X [N, D_in]
            grad_W [K, D_out, D_in]
        """
        X, W, a = ctx.for_backward #ctx.saved_tensors
        grad_X = grad_W = grad_a = None
        if ctx.needs_input_grad[0]:
            grad_X = torch.zeros_like(X)
            #W_T = W.transpose(1, 2).contiguous()
            #sparse_kpconv_gpu(grad_Y, W_T, a, grad_X)

        if ctx.needs_input_grad[1]:
            grad_W = torch.zeros_like(W)
            #tensor_outer_gpu(a, grad_Y, X, grad_W)

        assert not ctx.needs_input_grad[2], "alpha should not need gradient"

        return grad_X, grad_W, grad_a
        

sparse_kpconv_aggr = SparseKPConv.apply

if __name__ == '__main__':
    N = 100000
    K = 20
    D1 = 64
    D2 = 128 
    X = torch.nn.Parameter((torch.rand(N, D1)*1e-3).cuda(), requires_grad=True)
    W = torch.nn.Parameter((torch.rand(K, D2, D1)*1e-3).cuda(), requires_grad=True)
    #W = (torch.rand(K, D2, D1)*1e-3).cuda()
    optimizer = torch.optim.Adam([X, W], lr=1e-3)
    a = torch.rand(N, K).cuda() # randn(N, K).cuda()
    import time
    for itr in range(100):
        t0 = time.time()
        optimizer.zero_grad()
        loss = sparse_kpconv_aggr(X, W, a).square().sum() / 2.0
        loss.backward()
        optimizer.step()
        t4 = time.time()
        cost = float(torch.cuda.memory_allocated() / (1024 * 1024))
        print(f'iter={itr}, loss={loss.item()}, step={t4-t0}, mem={cost}')
    #eps = 1e-4
    #for n in range(N):
    #    for i in range(D1):
    #        X.data[n, i] += eps
    #        Y1, l1 = loss(X, W, a)
    #        pred_delta = X.grad[n, i]
    #        gt_delta = (l1 - l0)/eps
    #        diff = (gt_delta - pred_delta).abs()
    #        assert diff < 1e-3, f'{diff}'
    #        print(f'pass, {pred_delta:.6f}, {gt_delta:.6f}')
    #        X.data[n, i] -= eps
    #
    #for k in range(K):
    #    for i in range(D2):
    #        for j in range(D1):
    #            W.data[k, i, j] += eps
    #            Y1, l1 = loss(X, W, a)
    #            pred_delta = W.grad[k, i, j]
    #            gt_delta = (l1 - l0)/eps
    #            diff = (gt_delta - pred_delta).abs()
    #            try:
    #                assert diff < 1e-3, f'{diff}, {pred_delta}, {gt_delta}'
    #            except Exception as e:
    #                import ipdb; ipdb.set_trace()
    #                print(e)
    #            print(f'pass, {pred_delta:.6f}, {gt_delta:.6f}')
    #            W.data[k, i, j] -= eps
    #        
