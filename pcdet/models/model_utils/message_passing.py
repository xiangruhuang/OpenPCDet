import torch
from torch import nn
from torch_scatter import scatter
from torch.autograd import Function
import numpy as np
from tqdm import tqdm
from pcdet.ops.torch_hash.torch_hash_modules import (
    RadiusGraph,
)
import torch_cluster
import dgl

BATCHSIZE=4096
BUFFERSIZE=2**26

def dist2weight(dist):
    edge_weight = 1.0 / (dist + 1e-5) # [E, act_k]
    edge_weight_sum = edge_weight.sum(-1, keepdim=True) # [E, 1]
    edge_weight = edge_weight / edge_weight_sum # [E, act_k]

    return edge_weight

def get_batch_size(S):
    batchsize = BATCHSIZE
    while (batchsize > 1) and (batchsize * S > BUFFERSIZE):
        batchsize = batchsize // 2
    return batchsize

class MessagePassing(Function):
    @staticmethod
    def forward(ctx, kernel_weights, kernel_pos, ref_bxyz, ref_feat, query_bxyz, e_ref, e_query, num_act_kernels):
        """Compute message passing, save memory.
        Args:
            kernel_weights [K, D1, D2]
            kernel_pos [K, 3]
            ref_bxyz [N, 4]
            ref_feat [N, D1]
            query_bxyz [M, 4]
            e_ref [E]
            e_query [E]
            num_act_kernels: act_k
        Returns:
            query_feat [M, D2]
        """

        # compute edge weights
        pos_diff = (ref_bxyz[e_ref] - query_bxyz[e_query])[:, 1:4] # [E, act_k]
        e_edge, e_kernel = torch_cluster.knn(kernel_pos, pos_diff, num_act_kernels) # [E*act_k], [E*act_k]
        e_edge, e_kernel = e_edge.view(-1, num_act_kernels), e_kernel.view(-1, num_act_kernels) # [E, act_k]
        dist = (pos_diff[e_edge] - kernel_pos[e_kernel]).norm(p=2, dim=-1) # [E, act_k]
        weight = dist2weight(dist)

        # divide edges into blocks, grouped by e_query
        num_queries = query_bxyz.shape[0]
        batchsize = get_batch_size(e_query.shape[0] // num_queries * max(kernel_weights.shape[1], kernel_weights.shape[2]))
        num_batches = (num_queries + batchsize - 1) // batchsize # M // batchsize

        query_feat = []
        
        for b in range(num_batches):
            start = b * batchsize
            end = min((b + 1) * batchsize, num_queries)
            
            # dealing with these edges for now
            mask = (e_query >= start) & (e_query < end) # [E], E_b nonzeros
            
            e_edge_b = e_edge[mask, :] # [E_b, act_k]
            e_kernel_b = e_kernel[mask, :] # [E_b, act_k]
            weight_b = weight[mask, :] # [E_b, act_k]

            e_query_b = e_query[mask] # [E_b]
            e_ref_b = e_ref[mask] # [E_b]

            edge_feat_b = ref_feat.new_zeros(e_ref_b.shape[0], kernel_weights.shape[-1])
            for g in range(num_act_kernels):
                e_kernel_bg = e_kernel_b[:, g]
                if True:
                    edge_feat_bg = dgl.ops.gather_mm(ref_feat[e_ref_b],
                                                     kernel_weights,
                                                     idx_b = e_kernel_bg
                                                     ) * weight_b[:, g:(g+1)]
                    if g == 0:
                        edge_feat_b = edge_feat_bg
                    else:
                        edge_feat_b += edge_feat_bg
                else:
                    kernel_degree = scatter(torch.ones_like(e_kernel_bg), e_kernel_bg, dim=0,
                                            dim_size=kernel_weights.shape[0], reduce='sum')
                    sorted_kernel, reverse_idx = torch.sort(e_kernel_bg)
                    #import ipdb; ipdb.set_trace() # verify if indexing is correct
                    edge_feat_b[reverse_idx] += dgl.ops.segment_mm(ref_feat[e_ref_b[reverse_idx]],
                                                                  kernel_weights,
                                                                  seglen_a=kernel_degree.cpu()
                                                                  ) * weight_b[reverse_idx, g:(g+1)]

            
            #kernel_weights_b = kernel_weights[e_kernel_b] # [E_b, act_k, D1, D2]
            #kernel_weights_b = (kernel_weights_b * weight_b[:, :, None, None]).sum(dim=1) # [E_b, D1, D2]
        
            #edge_feat_b = (ref_feat[e_ref_b].unsqueeze(-2) @ kernel_weights_b).squeeze(-2) # [E_b, 1, D1] @ [E_b, D1, D2] = [E_b, 1, D2]

            #edge_feat_b = ref_feat[e_ref_b].unsqueeze(-2) @ kernel_weights[e_kernel_b] # [E_b, act_k, 1, D1] @ [E_b, act_k, D1, D2] = [E_b, act_k, 1, D2]
            
            #edge_feat_b = (edge_feat_b.squeeze(-2) * weight_b.unsqueeze(-1)).sum(dim=1) # [E_b, D2]
            
            query_feat_b = scatter(edge_feat_b, e_query_b-start,
                                   dim=0, dim_size=end-start, reduce='sum') # [B, D2]

            query_feat.append(query_feat_b)

        query_feat = torch.cat(query_feat, dim=0) # [M, D2]
        
        ctx.save_for_backward(kernel_weights, kernel_pos, ref_bxyz, ref_feat, 
                              query_bxyz, e_ref, e_query, weight, e_edge, e_kernel)

        return query_feat

    @staticmethod
    def backward(ctx, grad_query_feat):
        """
        Args:
            grad_query_feat [M, D2]

        Returns:
            grad_mlp [D1, D2]
            grad_ref_feat [N, D1]
        """
        kernel_weights, kernel_pos, ref_bxyz, ref_feat, \
            query_bxyz, e_ref, e_query, weight, e_edge, e_kernel = ctx.saved_tensors
        num_refs = ref_bxyz.shape[0]
        batchsize = get_batch_size(e_ref.shape[0] // num_refs * max(kernel_weights.shape[1], kernel_weights.shape[2]))
        num_batches = (num_refs + batchsize - 1) // batchsize # M // batchsize
        num_act_kernels = weight.shape[-1]
        D1 = ref_feat.shape[-1]
        D2 = grad_query_feat.shape[-1]
        K = kernel_weights.shape[0]

        grad_kernel_weights = torch.zeros(K, D1, D2).to(kernel_weights.device)
        grad_ref_feat = [] #torch.zeros(ref_feat.shape[0], D1)
        
        num_kernels = kernel_weights.shape[-1]
        for g in range(num_act_kernels):
            e_kernel_g = e_kernel[:, g] # [E]
            for k in range(num_kernels):
                mask = (e_kernel_g == k)
                if mask.any():
                    grad_kernel_weights[k] += (weight[mask, g] * ref_feat[e_ref[mask]].T) @ grad_query_feat[e_query[mask]] # [D1, D2]

        for b in range(num_batches):
            start = b * batchsize
            end = min((b + 1) * batchsize, num_refs)
            mask = (e_ref >= start) & (e_ref < end) # [E_b]

            e_edge_b = e_edge[mask, :] # [E_b, act_k]
            e_kernel_b = e_kernel[mask, :] # [E_b, act_k]
            weight_b = weight[mask, :] # [E_b, act_k]
            e_query_b = e_query[mask] # [E_b]
            e_ref_b = e_ref[mask] # [E_b]
            
            # interpolate kernel weights
            for g in range(num_act_kernels):
                grad_edge_feat_bg = dgl.ops.gather_mm(grad_query_feat[e_query_b], # [E, D2]
                                                      kernel_weights.transpose(1, 2), # [K, D2, D1]
                                                      idx_b = e_kernel_b[:, g]
                                                      ) * weight_b[:, g:(g+1)]
                if g == 0:
                    grad_edge_feat_b = grad_edge_feat_bg
                else:
                    grad_edge_feat_b += grad_edge_feat_bg
            #kernel_weights_b = kernel_weights[e_kernel_b] # [E_b, act_k, D1, D2]
            #kernel_weights_b = (kernel_weights_b * weight_b[:, :, None, None]).sum(dim=1) # [E_b, D1, D2]

            #grad_edge_feat_b = (kernel_weights_b @ grad_query_feat[e_query_b].unsqueeze(-1)).squeeze(-1) # [E_b, D1, 1]
            
            grad_ref_feat_b = scatter(grad_edge_feat_b, e_ref_b-start, dim=0,
                                      dim_size=end-start, reduce='sum') # [B, D1]

                
            grad_ref_feat.append(grad_ref_feat_b)

        grad_ref_feat = torch.cat(grad_ref_feat, dim=0)

        return grad_kernel_weights, None, None, grad_ref_feat, None, None, None, None

message_passing = MessagePassing.apply


# this function is only used for debug
def message_passing_naive(kernel_weights, kernel_pos, ref_bxyz, ref_feat, query_bxyz, e_ref, e_query, num_act_kernels):
    """Compute message passing, save memory.
    Args:
        kernel_weights [K, D1, D2]
        kernel_pos [K, 3]
        ref_bxyz [N, 4]
        ref_feat [N, D1]
        query_bxyz [M, 4]
        e_ref [E]
        e_query [E]
        num_act_kernels: act_k
    Returns:
        query_feat [M, D2]
    """
    # compute edge weights
    pos_diff = (ref_bxyz[e_ref] - query_bxyz[e_query])[:, 1:4] # [E, act_k]
    e_edge, e_kernel = torch_cluster.knn(kernel_pos, pos_diff, num_act_kernels) # [E*act_k], [E*act_k]
    e_edge, e_kernel = e_edge.view(-1, num_act_kernels), e_kernel.view(-1, num_act_kernels) # [E, act_k]
    dist = (pos_diff[e_edge] - kernel_pos[e_kernel]).norm(p=2, dim=-1) # [E, act_k]
    weight = dist2weight(dist) # [E, act_k]

    query_feat = None
    
    kernel_weights_ = []

    for g in range(weight.shape[-1]):
        e_kernel_g = e_kernel[:, g] # [E]
        edge_feat_g = dgl.ops.gather_mm(ref_feat[e_ref],
                                        kernel_weights,
                                        idx_b = e_kernel_g,
                                        ) * weight[:, g:(g+1)]
        if g == 0:
            edge_feat = edge_feat_g
        else:
            edge_feat += edge_feat_g
        #kernel_weights_.append(kernel_weights[e_kernel_g] * weight_g[:, None, None]) # [E, D1, D2] * [E]
    #edge_kernel_weights = torch.stack(kernel_weights_, dim=0).sum(0) # [E, D1, D2]
    #edge_feat = (ref_feat[e_ref].unsqueeze(-2) @ edge_kernel_weights).squeeze(1) # ([E, 1, D1] @ [E, D1, D2]).squeeze(1) = [E, D2]
    query_feat = scatter(edge_feat, e_query, dim=0, dim_size = query_bxyz.shape[0], reduce='sum')

    return query_feat

if __name__ == '__main__':
    #msg = MessagePassing(32, 128, 20).cuda()
    #feat = torch.randn(1500, 32).cuda()
    #e_query = torch.cat([torch.arange(1500), torch.arange(1500), torch.arange(1500)]).long().cuda()
    #e_ref = torch.cat([torch.arange(1500), torch.arange(1500) + 1, torch.arange(1500) + 2]) % 1500
    #e_ref = e_ref.long().cuda()

    #y = msg(bxyz, feat, bxyz, e_ref, e_query)
    #loss = y.sum()
    #loss.backward()

    channels = [16, 32]
    mlp = nn.Parameter(torch.randn(10, channels[0], channels[-1]).double().cuda(), requires_grad=True)
    mlp_pos = torch.randn(10, 3).double().cuda()

    ref_bxyz = torch.randn(1024, 4).double().cuda()
    ref_bxyz[:, 0] = torch.randint(size=[1024], high=2).double().cuda()
    query_bxyz = ref_bxyz
    ref_feat = torch.nn.Parameter(torch.randn(1024, channels[0]).cuda().double(), requires_grad=True)
    e_ref = torch.arange(1024).repeat(10).long().cuda()
    e_query = torch.from_numpy(np.random.permutation(10240) % 1024).long().cuda()
    memory_before = torch.cuda.memory_allocated() / 2**30
    query_feat = message_passing(mlp, mlp_pos, ref_bxyz, ref_feat, query_bxyz, e_ref, e_query, 3)
    if True:
        query_feat2 = message_passing_naive(mlp, mlp_pos, ref_bxyz, ref_feat, query_bxyz, e_ref, e_query, 3)
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
                query_feat1 = message_passing(mlp, mlp_pos, ref_bxyz, ref_feat, query_bxyz, e_ref, e_query, 3)
                loss1 = query_feat1.sum()
                grad_ij = (loss1 - loss) / eps
                assert (grad_ij - grad_mlp[k, i, j]).abs() < 1e-4, f"{grad_ij}, {grad_mlp[i, j]}"
                mlp[k, i, j].data -= eps
    print('done with mlp testing')

    for i in tqdm(range(ref_feat.shape[0])):
        for j in tqdm(range(ref_feat.shape[1])):
            ref_feat.data[i, j] += eps
            query_feat1 = message_passing(mlp, mlp_pos, ref_bxyz, ref_feat, query_bxyz, e_ref, e_query, 3)
            loss1 = query_feat1.sum()
            grad_ij = (loss1 - loss) / eps
            assert (grad_ij - grad_ref_feat[i, j]).abs() < 1e-4, f"{grad_ij}, {grad_ref_feat[i, j]}"
            ref_feat.data[i, j] -= eps

