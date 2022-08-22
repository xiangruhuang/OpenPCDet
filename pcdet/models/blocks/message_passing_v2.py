import numpy as np
import torch
from torch import nn
from torch_scatter import scatter
from torch.autograd import Function
from tqdm import tqdm


def kernel_dist(index, K0, D1, D2, E):
    """
    Args:
        index [E]
        K0, D1, D2, E: integers
    Returns:
        
    """
    index = index.long()
    degree = scatter(torch.ones_like(index), index, dim=0,
                     dim_size=K0, reduce='sum')
    kernel_offset = degree.cumsum(dim=0) - degree
    
    # find best batch size B, [K0, D1, D2]->[K, D1, D2]
    best_batch_size = -1
    min_cost = 1e10
    _batch_size = 1
    max_degree = degree.max().long().item()
    while True:
        batch_size = np.round(_batch_size).astype(np.int64)
        if batch_size > max_degree:
            break
        num_duplicate_kernels = torch.ceil(degree / batch_size).clamp(min=1).sum().item()
        
        mem_cost = num_duplicate_kernels * D1 * D2 + num_duplicate_kernels * batch_size * (D1 + D2)
        compute_cost = num_duplicate_kernels * batch_size * D1 * D2
        cost = mem_cost + 0.01 * compute_cost
        if cost < min_cost:
            min_cost = cost
            best_batch_size = batch_size
        _batch_size *= 1.5

    B = best_batch_size
    num_duplicates = torch.ceil(degree / B).long().clamp(min=1) # [1, 1, 2, 1]
    offset = num_duplicates.cumsum(dim=0) - num_duplicates # [1, 2, 4, 5] - [1, 1, 2, 1] = [0, 1, 2, 4]
    K = num_duplicates.sum().long().item() # number of kernels (include duplicate)
    
    # [0,1,2,4] -> [0, 1, 1, 0, 1].cumsum() -> [0, 1, 2, 2, 3]
    original_kernel_index = index.new_zeros(K)
    original_kernel_index[offset[1:]] = 1
    original_kernel_index = original_kernel_index.cumsum(dim=0)
    
    # [E] -> [K, B]
    unique_index, inverse = torch.sort(index)
    global_offset = index.new_zeros(E)
    global_offset[inverse] = torch.arange(E).to(index)
    pool_offset = (global_offset - kernel_offset[index]) + offset[index] * B
    
    # map from [K0] to [K]
    
    return original_kernel_index, pool_offset, K, B

def message_passing_naive(kernel, ref_feat, e_kernel, e_ref, e_query, num_queries):
    import dgl
    edge_feat = dgl.ops.gather_mm(ref_feat[e_ref], kernel, idx_b = e_kernel.int())
        
    query_feat = scatter(edge_feat, e_query, dim=0,
                         dim_size=num_queries, reduce='sum')

    return query_feat

class MessagePassing(Function):

    @staticmethod
    def forward(ctx, kernel, ref_feat, e_kernel, e_ref, e_query, num_queries):
        """
        Args:
            kernel [K, D1, D2]: kernel weights
            ref_feat [N, D1]: source features
            e_kernel [E]: (in range [K]) edge associated kernel index
            e_ref [E]: (in range [N]) edge source endpoints
            e_query [E]: (in range [M]) edge target endpoints

        Returns:
            query_feat: [M, D2] 
        """
        K0, D1, D2 = list(kernel.shape)
        E = e_kernel.shape[0]
        original_kernel_index, pool_index, K, B = \
                kernel_dist(e_kernel, K0, D1, D2, E)
        dup_kernel = kernel[original_kernel_index] # [K0, D1, D2]
        pool = ref_feat.new_zeros(K*B, D1)
        pool[pool_index] = ref_feat[e_ref]
        pool = pool.view(K, B, -1) # [K0, B, D1]
        query_pool = pool @ dup_kernel # [K0, B, D2]
        query_edge_feat = query_pool.view(K*B, -1)[pool_index]
        
        query_feat = scatter(query_edge_feat, e_query, dim=0,
                             dim_size=num_queries, reduce='sum')

        ctx.save_for_backward(kernel, ref_feat, e_kernel, e_ref, e_query)

        return query_feat

    @staticmethod
    def backward(ctx, grad_query_feat):
        """
        Args:
            grad_query_feat [M, D2] gradient of query features

        Returns:
            grad_ref_feat [N, D1] gradient to ref features
            grad_kernel [K, D1, D2] gradient to kernel weights
        """
        kernel, ref_feat, e_kernel, e_ref, e_query = ctx.saved_tensors
        num_refs = ref_feat.shape[0]

        K0, D1, D2 = list(kernel.shape)
        E = e_kernel.shape[0]

        original_kernel_index, pool_index, K, B = \
                kernel_dist(e_kernel, K0, D2, D1, E)
        
        dup_kernel = kernel[original_kernel_index].transpose(1, 2) # [K0, D2, D1]
        pool = grad_query_feat.new_zeros(K*B, D2)
        pool[pool_index] = grad_query_feat[e_query]
        pool = pool.view(K, B, -1) # [K0, B, D2]

        grad_ref_pool = pool @ dup_kernel # [K0, B, D1]
        grad_ref_edge_feat = grad_ref_pool.view(K*B, -1)[pool_index]
        
        grad_ref_feat = scatter(grad_ref_edge_feat, e_ref, dim=0,
                                dim_size=num_refs, reduce='sum')
        
        # compute gradient w.r.t. kernel
        pool_ref = grad_query_feat.new_zeros(K*B, D1)
        pool_ref[pool_index] = ref_feat[e_ref] # [K0, B, D1]
        pool_ref = pool_ref.view(K, B, -1).transpose(1, 2) # [K0, D1, B]
        grad_dup_kernel = pool_ref @ pool # [K0, D1, D2]
        grad_kernel = scatter(grad_dup_kernel, original_kernel_index, dim=0,
                              dim_size=K0, reduce='sum')

        return grad_kernel, grad_ref_feat, None, None, None


message_passing = MessagePassing.apply

if __name__ == '__main__':
    #msg = MessagePassing(32, 128, 20).cuda()
    #feat = torch.randn(1500, 32).cuda()
    #e_query = torch.cat([torch.arange(1500), torch.arange(1500), torch.arange(1500)]).long().cuda()
    #e_ref = torch.cat([torch.arange(1500), torch.arange(1500) + 1, torch.arange(1500) + 2]) % 1500
    #e_ref = e_ref.long().cuda()

    #y = msg(bxyz, feat, bxyz, e_ref, e_query)
    #loss = y.sum()
    #loss.backward()

    d1 = 16
    d2 = 32
    K = 10
    N = 400
    deg = 10

    channels = [d1, d2]
    mlp = nn.Parameter(torch.randn(K, channels[0], channels[-1]).double().cuda(), requires_grad=True)

    ref_feat = torch.nn.Parameter(torch.randn(N, d1).cuda().double(), requires_grad=True)
    e_ref = torch.arange(N).repeat(deg).long().cuda()
    e_query = torch.from_numpy(np.random.permutation(N*deg) % N).long().cuda()
    e_kernel = torch.arange(N*deg).long().cuda() % K
    query_feat = message_passing(mlp, ref_feat, e_kernel, e_ref, e_query)

    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        query_feat = message_passing(mlp, ref_feat, e_kernel, e_ref, e_query)
    print(prof.key_averages().table(sort_by="self_cuda_time_total"))
    if True:
        with torch.autograd.profiler.profile(use_cuda=True) as prof2: 
            query_feat2 = message_passing_naive(mlp, ref_feat, e_kernel, e_ref, e_query)
        print(prof2.key_averages().table(sort_by="self_cuda_time_total"))
        assert (query_feat - query_feat2).abs().max() < 1e-5
    
    loss = query_feat.sum()
    loss.backward()

    grad_mlp = mlp.grad
    grad_ref_feat = ref_feat.grad
    eps = 1e-5
    for k in tqdm(range(mlp.shape[0])):
        for i in tqdm(range(mlp.shape[0])):
            for j in tqdm(range(mlp.shape[1])):
                mlp[k, i, j].data += eps
                #query_feat1 = message_passing(mlp, mlp_pos, ref_bxyz, ref_feat, query_bxyz, e_ref, e_query, 3)
                query_feat1 = message_passing(mlp, ref_feat, e_kernel, e_ref, e_query)
                loss1 = query_feat1.sum()
                grad_ij = (loss1 - loss) / eps
                assert (grad_ij - grad_mlp[k, i, j]).abs() < 1e-4, f"{grad_ij}, {grad_mlp[i, j]}"
                mlp[k, i, j].data -= eps
    print('done with mlp testing')

    for i in tqdm(range(0, ref_feat.shape[0], 10)):
        for j in tqdm(range(ref_feat.shape[1])):
            ref_feat.data[i, j] += eps
            #query_feat1 = message_passing(mlp, mlp_pos, ref_bxyz, ref_feat, query_bxyz, e_ref, e_query, 3)
            query_feat1 = message_passing(mlp, ref_feat, e_kernel, e_ref, e_query)
            loss1 = query_feat1.sum()
            grad_ij = (loss1 - loss) / eps
            assert (grad_ij - grad_ref_feat[i, j]).abs() < 1e-4, f"{grad_ij}, {grad_ref_feat[i, j]}"
            ref_feat.data[i, j] -= eps

    print('done with ref_feat testing')
