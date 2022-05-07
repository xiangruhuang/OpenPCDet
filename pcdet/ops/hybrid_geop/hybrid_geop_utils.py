import torch
from hybrid_geop_cuda import (
    svd3_gpu
)
    
def hybrid_geop_fitting(points, edge_indices):
    """Fit hybrid geometric primitives to point cloud.
    Args:
        points [N, 4]
        edge_indices [2, E]

    Returns:
        
    """
    import ipdb; ipdb.set_trace()
    pass

if __name__ == '__main__':
    A = torch.randn(10, 3, 3).cuda()
    U = torch.empty(10, 3, 3).cuda()
    S = torch.empty(10, 3).cuda()
    VT = torch.empty(10, 3, 3).cuda()
    svd3_gpu(A, U, S, VT)
    U1, S1, V1 = A.svd()
    import ipdb; ipdb.set_trace()
    pass
