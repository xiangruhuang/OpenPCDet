import torch
from torch import nn
import torch.nn.functional as F
import torch_cluster
from torch_scatter import scatter

def build_norm_layer(num_features, norm_cfg):
    norm_type = norm_cfg['type']
    norm_layer = getattr(nn, norm_type)
    
    norm_cfg_clone = {}
    norm_cfg_clone.update(norm_cfg)
    norm_cfg_clone.pop('type')
    norm_cfg_clone['num_features'] = num_features
    return norm_layer(**norm_cfg_clone)

def build_conv_layer(in_channels, out_channels, conv_cfg, **conv_kwargs):
    conv_type = conv_cfg['type']
    conv_layer = getattr(nn, conv_type)
    
    conv_cfg_clone = {}
    conv_cfg_clone.update(conv_cfg)
    conv_cfg_clone.pop('type')

    return conv_layer(in_channels=in_channels,
                      out_channels=out_channels,
                      **conv_cfg_clone,
                      **conv_kwargs)

def MLP(channels, activation=nn.LeakyReLU(0.2), bn_momentum=0.1, bias=True, eps=1e-5):
    return nn.Sequential(
        *[
            nn.Sequential(
                nn.Linear(channels[i - 1], channels[i], bias=bias),
                nn.BatchNorm1d(channels[i], momentum=bn_momentum, eps=eps),
                activation,
            )
            for i in range(1, len(channels))
        ]
    )

def MLPBlock(in_channel, out_channel, norm_cfg,
             activation=nn.LeakyReLU(0.2), bias=True):
    norm_layer = build_norm_layer(out_channel, norm_cfg)
    return nn.Sequential(
                nn.Linear(in_channel, out_channel, bias=bias),
                norm_layer,
                activation
           )

class PointConv(nn.Module):
    def __init__(self, input_channel, output_channel, pos_encoder_cfg):
        super(PointConv, self).__init__()
        pos_encoder_cfg_type = pos_encoder_cfg["TYPE"] 
        self.pos_encoder_cfg_type = pos_encoder_cfg_type
        if pos_encoder_cfg_type == 'linear':
            self.pos_encoder = nn.Linear(3, output_channel)
            self.norm = nn.BatchNorm1d(output_channel)
        elif pos_encoder_cfg_type == 'interpolate':
            self.num_kernel_points = pos_encoder_cfg.get("NUM_KERNEL_POINTS", 16)
            self.num_act_kernels = pos_encoder_cfg.get("NUM_ACT_KERNELS", 3)
            self.div_factor = pos_encoder_cfg.get("DIV_FACTOR", 1.0)

            X = Y = Z = torch.linspace(-1, 1, 100)
            
            candidate_points = torch.stack(torch.meshgrid(X, Y, Z), dim=-1).reshape(-1, 3)
            candidate_mask = candidate_points.norm(p=2, dim=-1) <= 1.0
            candidate_points = candidate_points[candidate_mask]

            ratio = (self.num_kernel_points + 1) / candidate_points.shape[0]
            kernel_pos_index = torch_cluster.fps(candidate_points, None, ratio,
                                                 random_start=True)[:self.num_kernel_points]
            kernel_pos = candidate_points[kernel_pos_index]
            self.register_buffer('kernel_pos', kernel_pos, persistent=True)

            self.kernel_embedding = nn.Parameter(torch.randn(self.num_kernel_points,
                                                             output_channel)/(output_channel**0.5),
                                                 requires_grad=True)

            #self.pos_encoder = partial(
            #                       PosEmbedding.apply,
            #                       kernel_pos=self.kernel_pos,
            #                       kernel_embedding=self.kernel_embedding,
            #                   )

        self.mlp = nn.Linear(input_channel, output_channel)
        self.aggr = 'mean'

    def forward(self, ref_bxyz, ref_feat, query_bxyz, e_ref, e_query):
        """
        Args:
            ref_bxyz [N, 4]
            ref_feat [N, C]
            query_bxyz [M, 4]
            e_ref, e_query: point index of (ref, query), representing edges
        Returns:
            query_feat [M, C_out]
        """
        pos_diff = (ref_bxyz[e_ref] - query_bxyz[e_query])[:, 1:4]
        num_queries = query_bxyz.shape[0]
        
        if self.pos_encoder_cfg_type == 'linear':
            pos_embedding = self.pos_encoder(pos_diff) # [E, C_out]
            output_feat = self.mlp(ref_feat)[e_ref] # [E, C_out]
            query_feat = scatter(pos_embedding, e_query, dim=0, dim_size=num_queries, reduce=self.aggr)
            query_feat += scatter(output_feat, e_query, dim=0, dim_size=num_queries, reduce=self.aggr)
            query_feat = F.relu(self.norm(query_feat), inplace=False)
        elif self.pos_encoder_cfg_type == 'interpolate':
            import ipdb; ipdb.set_trace()
            with torch.no_grad():
                e_diff, e_kernel = torch_cluster.knn(self.kernel_pos, pos_diff, self.num_act_kernels)
                edge_dist = (self.kernel_pos[e_kernel] - pos_diff[e_diff]).norm(p=2, dim=-1)
                edge_weight = (-(edge_dist/self.div_factor).square()).exp()
                query_weight = scatter(edge_weight, e_diff, dim=0, dim_size=num_queries, reduce='sum')
                diff_pos_embedding = scatter(self.kernel_embedding[e_kernel], e_diff, dim=0,
                                             dim_size=num_queries, reduce='sum')
                diff_pos_embedding = diff_pos_embedding / diff_weight[:, None]

            pass

        return query_feat
            

        
        
        

