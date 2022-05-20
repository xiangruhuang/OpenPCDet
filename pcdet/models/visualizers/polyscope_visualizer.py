import torch
from torch import nn
import numpy as np
from collections import defaultdict
import polyscope as ps

def to_numpy(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    elif isinstance(a, np.ndarray):
        return a
    else:
        raise ValueError("Requiring Numpy or torch.Tensor")

class PolyScopeVisualizer(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.enabled = model_cfg.get('ENABLED', False)
        if self.enabled:
            self.point_cloud_vis = model_cfg.get("POINT_CLOUD", None)
            self.box_vis = model_cfg.get("BOX", None)
            self.graph_vis = model_cfg.get("GRAPH", None)
            self.primitive_vis = model_cfg.get("PRIMITIVE", None)
            self.shared_color_dict = model_cfg.get("SHARED_COLOR", None)
            self.output = model_cfg.get("OUTPUT", None)
            self.voxel_size = model_cfg.get('voxel_size', None)
            self.pc_range = model_cfg.get('pc_range', None)
            self.size_factor = model_cfg.get('size_factor', None)
            self.radius = model_cfg.get('radius', 2e-4)
            self.ground_plane = model_cfg.get("ground_plane", False)
            self.init()
    
    def color(self, color_name):
        if not hasattr(self, "_shared_color"):
            raise ValueError("Color Dictionary not initialized")
        return self._shared_color[color_name]

    def init(self):
        ps.set_up_dir('z_up')
        ps.init()
        if not self.ground_plane:
            ps.set_ground_plane_mode('none')
        if self.shared_color_dict is not None:
            color_dict = {}
            for color_name, color in self.shared_color_dict.items():
                if isinstance(color, list) and len(color) == 2:
                    color_dict[color_name] = np.random.uniform(size=color)
                else:
                    color_dict[color_name] = np.array(color)
            self._shared_color = color_dict

    def visualize(self, monitor=None):
        if monitor is None:
            return 
        if monitor == 'screen':
            self.show()
        elif isinstance(monitor, str):
            self.screenshot(monitor)
        else:
            raise ValueError(f"Unrecognized Monitor Option {monitor}")

    def forward(self, batch_dict):
        if not self.enabled:
            return

        for i in range(batch_dict['batch_size']):
            if self.point_cloud_vis is not None:
                for pc_key, vis_cfg_this in self.point_cloud_vis.items():
                    if pc_key not in batch_dict:
                        continue
                    vis_cfg = {}; vis_cfg.update(vis_cfg_this)
                    pointcloud = batch_dict[pc_key]
                    batch_key = vis_cfg.pop('batch') if 'batch' in vis_cfg else None
                    if batch_key is None:
                        batch_idx = pointcloud[:, 0]
                        pointcloud = pointcloud[:, 1:]
                    else:
                        batch_idx = batch_dict[batch_key][:, 0]
                    batch_mask = batch_idx == i
                    pointcloud = pointcloud[batch_mask, :3].detach().cpu()
                    if 'name' in vis_cfg:
                        pc_name = vis_cfg.pop('name')
                    else:
                        pc_name = pc_key
                    self.pointcloud(pc_name, pointcloud, batch_dict, batch_mask, **vis_cfg)
            
            if self.box_vis is not None:
                for box_key, vis_cfg_this in self.box_vis.items():
                    if box_key not in batch_dict:
                        continue
                    vis_cfg = {}; vis_cfg.update(vis_cfg_this)
                    boxes = batch_dict[box_key][i].detach().cpu()
                    labels = boxes[:, 7]
                    boxes = boxes[:, :7]
                    if 'name' in vis_cfg:
                        box_name = vis_cfg.pop('name')
                    else:
                        box_name = box_key
                    self.boxes_from_attr(box_name, boxes, labels, **vis_cfg)
            
            if self.graph_vis is not None:
                for graph_key, vis_cfg_this in self.graph_vis.items():
                    if graph_key not in batch_dict:
                        continue
                    vis_cfg = {}; vis_cfg.update(vis_cfg_this)
                    e_query, e_ref = batch_dict[graph_key].detach().cpu()
                    query_key = vis_cfg['query']
                    query_points = batch_dict[query_key]
                    ref_key = vis_cfg['ref']
                    ref_points = batch_dict[ref_key]

                    valid_mask = (query_points[e_query, 0].round().long() == i) & (ref_points[e_ref, 0].round().long() == i)
                    e_query, e_ref = e_query[valid_mask], e_ref[valid_mask]

                    # take this batch
                    query_batch_idx = torch.where(query_points[:, 0].round().long() == i)[0]
                    query_idx_map = torch.zeros(query_points.shape[0]).round().long().to(query_batch_idx.device)
                    query_idx_map[query_batch_idx] = torch.arange(query_batch_idx.shape[0]).to(query_idx_map)
                    query_points = query_points[query_batch_idx, 1:].detach().cpu()
                    e_query = query_idx_map[e_query]

                    ref_batch_idx = torch.where(ref_points[:, 0].round().long() == i)[0]
                    ref_idx_map = torch.zeros(ref_points.shape[0]).round().long().to(ref_batch_idx.device)
                    ref_idx_map[ref_batch_idx] = torch.arange(ref_batch_idx.shape[0]).to(ref_idx_map)
                    ref_points = ref_points[ref_batch_idx, 1:].detach().cpu()
                    e_ref = ref_idx_map[e_ref]
                
                    edge_indices = torch.stack([e_query, e_ref+query_points.shape[0]], dim=-1).detach().cpu()
                    
                    if 'name' in vis_cfg:
                        graph_name = vis_cfg.pop('name')
                    else:
                        graph_name = graph_key
                    all_points = torch.cat([query_points[:, :3], ref_points[:, :3]], dim=0)
                    self.curvenetwork(graph_name, all_points, edge_indices, batch_dict, valid_mask, **vis_cfg)

            if self.primitive_vis is not None:
                for primitive_key, vis_cfg in self.primitive_vis.items():
                    if primitive_key not in batch_dict:
                        continue
                    vis_cfg_this = {}; vis_cfg_this.update(vis_cfg)
                    primitives = batch_dict[primitive_key].detach().cpu()
                    batch_index = primitives[:, 0].round().long()
                    batch_mask = batch_index == i
                    primitives = primitives[batch_mask, 1:]
                    centers = primitives[:, :3]
                    cov = primitives[:, 5:14].view(-1, 3, 3)
                    S, R = torch.linalg.eigh(cov)
                    R = R * S[:, None, :].sqrt()
                    fitness = primitives[:, -1].view(-1)
                    corners = []
                    if False:
                        shell = torch.randn(20, 3)
                        shell = shell / shell.norm(p=2, dim=-1)[:, None]
                        point_balls = (R @ shell.T).transpose(1, 2) + centers[:, None, :]
                        point_balls = point_balls.reshape(-1, 3)
                        self.pointcloud(primitive_key, point_balls, None, None, **vis_cfg_this)
                    else:
                        for dx in [-1, 1]:
                            for dy, dz in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:
                                dvec = np.array([dx, dy, dz]).astype(np.float32)
                                corner = centers + (R * dvec).sum(-1)
                                corners.append(corner)
                        corners = torch.stack(corners, dim=1)
                        hexes = torch.arange(corners.shape[0]*8).view(-1, 8)
                        scalars = vis_cfg_this.pop("scalars") if "scalars" in vis_cfg else None
                        class_labels = vis_cfg_this.pop("class_labels") if "class_labels" in vis_cfg_this else None
                        ps_v = ps.register_volume_mesh(primitive_key, corners.view(-1, 3).detach().cpu().numpy(), hexes=hexes.numpy(), **vis_cfg_this)
                        ps_v.add_scalar_quantity('fitness', fitness.detach().cpu(), defined_on='cells')
                        if scalars:
                            for scalar_name, scalar_cfg in scalars.items():
                                ps_v.add_scalar_quantity('scalars/'+scalar_name, batch_dict[scalar_name][batch_mask].detach().cpu(), defined_on='cells', **scalar_cfg)
                        if class_labels:
                            for label_name, label_cfg in class_labels.items():
                                label = batch_dict[label_name][batch_mask].detach().cpu().long()
                                label_cfg_this = {}
                                for key, val in label_cfg.items():
                                    if (key == 'values') and isinstance(val, str):
                                        label_cfg_this[key] = self.color(val)[label]
                                        invalid_mask = label < 0
                                        label_cfg_this[key][invalid_mask] = np.array([75./255, 75./255, 75/255.])
                                    else:
                                        label_cfg_this[key] = val
                                ps_v.add_color_quantity('class_labels/'+label_name, defined_on='cells', **label_cfg_this)
                        
            self.visualize(monitor=self.output)

    def clear(self):
        ps.remove_all_structures()
        self.logs = []

    def pc_scalar(self, pc_name, name, quantity, enabled=False):
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        ps.get_point_cloud(pc_name).add_scalar_quantity(name, quantity, enabled=enabled)
    
    def pc_color(self, pc_name, name, color, enabled=False):
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        ps.get_point_cloud(pc_name).add_color_quantity(name, color, enabled=enabled)

    def corres(self, name, src, tgt):
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        points = torch.cat([src, tgt], dim=0)
        edges = torch.stack([torch.arange(src.shape[0]),
                             torch.arange(tgt.shape[0]) + src.shape[0]], dim=-1)
        return ps.register_curve_network(name, points, edges, radius=self.radius)

    def trace(self, name, points, **kwargs):
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        num_points = points.shape[0]
        edges = torch.stack([torch.arange(num_points-1),
                             torch.arange(num_points-1)+1], dim=-1)
        return ps.register_curve_network(name, points, edges, **kwargs)
   
    def curvenetwork(self, name, nodes, edges, data_dict, batch_mask, **kwargs):
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")

        edge_scalars = kwargs.pop("edge_scalars") if "edge_scalars" in kwargs else None
        radius = kwargs.get('radius', self.radius)
        ps_c = ps.register_curve_network(name, nodes, edges, radius=radius)

        if edge_scalars:
            for scalar_name, scalar_cfg in edge_scalars.items():
                scalar = data_dict[scalar_name][batch_mask].detach().cpu()
                ps_c.add_scalar_quantity('edge-scalars/'+scalar_name, scalar, defined_on='edges', **scalar_cfg)
        return ps_c


    def pointcloud(self, name, pointcloud, data_dict, batch_mask, color=None, radius=None, **kwargs):
        """Visualize non-zero entries of heat map on 3D point cloud.
            point cloud (torch.Tensor, [N, 3])
        """
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        if radius is None:
            radius = self.radius
        scalars = kwargs.pop("scalars") if "scalars" in kwargs else None
        class_labels = kwargs.pop("class_labels") if "class_labels" in kwargs else None

        if color is None:
            ps_p = ps.register_point_cloud(name, pointcloud, radius=radius, **kwargs)
        else:
            ps_p = ps.register_point_cloud(
                name, pointcloud, radius=radius, color=tuple(color), **kwargs
                )

        if scalars:
            for scalar_name, scalar_cfg in scalars.items():
                scalar = data_dict[scalar_name][batch_mask].detach().cpu()
                ps_p.add_scalar_quantity('scalars/'+scalar_name, scalar, **scalar_cfg)

        if class_labels:
            for label_name, label_cfg in class_labels.items():
                label = data_dict[label_name][batch_mask].detach().cpu().long()
                if label.shape[0] == 0:
                    continue
                label_cfg_this = {}
                for key, val in label_cfg.items():
                    if (key == 'values') and isinstance(val, str):
                        label_cfg_this[key] = self.color(val)[label]
                        invalid_mask = label < 0
                        label_cfg_this[key][invalid_mask] = np.array([75./255, 75./255, 75/255.])
                    else:
                        label_cfg_this[key] = val
                if label_cfg_this.get('values', None) is None:
                    print(label.shape, label_name)
                    label_cfg_this['values'] = np.random.randn(label.max()+100, 3)[label]
                ps_p.add_color_quantity('class_labels/'+label_name, **label_cfg_this)

        return ps_p
    
    def get_meshes(self, centers, eigvals, eigvecs):
        """ Prepare corners and faces (for visualization only). """

        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        v1 = eigvecs[:, :3]
        v2 = eigvecs[:, 3:]
        e1 = np.sqrt(eigvals[:, 0:1])
        e2 = np.sqrt(eigvals[:, 1:2])
        corners = []
        for d1 in [-1, 1]:
            for d2 in [-1, 1]:
                corners.append(centers + d1*v1*e1 + d2*v2*e2)
        num_voxels = centers.shape[0]
        corners = np.stack(corners, axis=1) # [M, 4, 3]
        faces = [0, 1, 3, 2]
        faces = np.array(faces, dtype=np.int32)
        faces = np.repeat(faces[np.newaxis, np.newaxis, ...], num_voxels, axis=0)
        faces += np.arange(num_voxels)[..., np.newaxis, np.newaxis]*4
        return corners.reshape(-1, 3), faces.reshape(-1, 4)
    
    def planes(self, name, planes):
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        corners, faces = self.get_meshes(planes[:, :3], planes[:, 6:8], planes[:, 8:14])
        return ps.register_surface_mesh(name, corners, faces)

    def boxes_from_attr(self, name, attr, labels=None, **kwargs):
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        from pcdet.utils.box_utils import boxes_to_corners_3d
        corners = boxes_to_corners_3d(attr)
        if 'with_ori' in kwargs:
            with_ori = kwargs.pop('with_ori')
        else:
            with_ori = False
        ps_box = self.boxes(name, corners, labels, **kwargs)
        #if with_ori:
        #    ori = attr[:, -1]
        #    sint, cost = np.sin(ori), np.cos(ori)
        #    arrow = np.stack([sint, cost, np.zeros_like(cost)], axis=-1)[:, np.newaxis, :].repeat(8, 1)
        #    ps_box.add_vector_quantity('orientation', arrow.reshape(-1, 3), enabled=True)
        

    def boxes(self, name, corners, labels=None, **kwargs):
        """
            corners (shape=[N, 8, 3]):
            labels (shape=[N])
        """
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        #  0    1
        #     3    2
        #  |    |
        #  4    5
        #     7    6
        #edges = [[0, 1], [0, 3], [0, 4], [1, 2],
        #         [1, 5], [2, 3], [2, 6], [3, 7],
        #         [4, 5], [4, 7], [5, 6], [6, 7]]
        N = corners.shape[0]
        #edges = np.array(edges) # [12, 2]
        #edges = np.repeat(edges[np.newaxis, ...], N, axis=0) # [N, 12, 2]
        #offset = np.arange(N)[..., np.newaxis, np.newaxis]*8 # [N, 1, 1]
        #edges = edges + offset
        #if kwargs.get('radius', None) is None:
        #    kwargs['radius'] = 2e-4
        corners = to_numpy(corners)
        corners = corners.reshape(-1, 3)
        ps_box = ps.register_volume_mesh(
                    name, corners,
                    hexes=np.arange(corners.shape[0]).reshape(-1, 8),
                    **kwargs
                 )
        ps_box.set_transparency(0.2)
        if labels is not None:
            # R->Car, G->Ped, B->Cyc
            colors = np.array([[1,0,0], [1,0,0], [0,1,0], [0,0,1], [1,0,1], [1,1,0]])
            labels = to_numpy(labels).astype(np.int64) 
            #labels = np.repeat(labels[:, np.newaxis], 8, axis=-1).reshape(-1).astype(np.int64)
            ps_box.add_color_quantity('class', colors[labels], defined_on='cells', enabled=True)
            ps_box.add_scalar_quantity('scalars/class', labels, defined_on='cells')

        return ps_box

    def wireframe(self, name, heatmap):
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        size_y, size_x = heatmap.shape
        x, y = torch.meshgrid(heatmap)
        return x, y

    def heatmap(self, name, heatmap, color=True, threshold=0.1,
                **kwargs):
        """Visualize non-zero entries of heat map on 3D point cloud.
        `voxel_size`, `size_factor`, `pc_range` need to be specified.
        By default, the heatmap need to be transposed.

        Args:
            heatmap (torch.Tensor or np.ndarray, [W, H])

        """
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        if isinstance(heatmap, np.ndarray):
            heatmap = torch.from_numpy(heatmap)

        if self.voxel_size is None:
            raise ValueError("self.voxel_size not specified")
        
        heatmap = heatmap.T
        size_x, size_y = heatmap.shape
        x, y = torch.meshgrid(torch.arange(size_x),
                              torch.arange(size_y),
                              indexing="ij")
        x, y = x.reshape(-1), y.reshape(-1)
        z = heatmap.reshape(-1)

        mask = torch.zeros(size_x+2, size_y+2, size_x+2, size_y+2, dtype=torch.bool)
        
        for dx, dy in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
            mask[x+1, y+1, x+1+dx, y+1+dy] = True
        x0, y0, x1, y1 = torch.where(mask)
        x0, y0, x1, y1 = x0-1, y0-1, x1-1, y1-1
        is_inside = ((x1 >= size_x) | (x1 < 0) | (y1 >= size_y) | (y1 < 0)) == False
        e0 = (x0 * size_y + y0)[is_inside]
        e1 = (x1 * size_y + y1)[is_inside]
        
        edges = torch.stack([e0, e1], dim=-1)
        x = x * self.size_factor * self.voxel_size[0] + self.pc_range[0]
        y = y * self.size_factor * self.voxel_size[1] + self.pc_range[1]
        nodes = torch.stack([x, y, z], dim=-1)
        radius = kwargs.get("radius", self.radius*10)
        ps_c = self.curvenetwork(name, nodes, edges, radius=radius)
        
        if color:
            ps_c.add_scalar_quantity("height", z, enabled=True) 

        return ps_c

    def show(self):
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        ps.set_up_dir('z_up')
        ps.init()
        ps.show()

    def look_at(self, center, distance=100, bev=True, **kwargs):
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        if bev:
            camera_loc = center + np.array([0, 0, distance])
            # look down from bird eye view
            # with +y-axis being the up dir on the image
            ps.look_at_dir(camera_loc, center, (0,1,0), **kwargs)
        else:
            raise ValueError("Not Implemented Yet, please use bev=True")

    def screenshot(self, filename, **kwargs):
        if not self.enabled:
            raise ValueError(f"Visualizer {self.__class__} is not Enabled")
        ps.screenshot(filename, **kwargs)
