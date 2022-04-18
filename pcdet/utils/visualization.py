import polyscope as ps
import torch
import numpy as np

class Visualizer:
    def __init__(self,
                 voxel_size=None,
                 pc_range=None,
                 size_factor=None,
                 ground_plane=False,
                 radius=2e-4):
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.size_factor = size_factor
        self.radius = radius
        ps.set_up_dir('z_up')
        ps.init()
        if not ground_plane:
            ps.set_ground_plane_mode('none')
            
        self.logs = []

    def clear(self):
        ps.remove_all_structures()
        self.logs = []

    def pc_scalar(self, pc_name, name, quantity, enabled=False):
        ps.get_point_cloud(pc_name).add_scalar_quantity(name, quantity, enabled=enabled)
    
    def pc_color(self, pc_name, name, color, enabled=False):
        ps.get_point_cloud(pc_name).add_color_quantity(name, color, enabled=enabled)

    def corres(self, name, src, tgt):
        points = torch.cat([src, tgt], dim=0)
        edges = torch.stack([torch.arange(src.shape[0]),
                             torch.arange(tgt.shape[0]) + src.shape[0]], dim=-1)
        return ps.register_curve_network(name, points, edges, radius=self.radius)

    def trace(self, name, points, **kwargs):
        num_points = points.shape[0]
        edges = torch.stack([torch.arange(num_points-1),
                             torch.arange(num_points-1)+1], dim=-1)
        return ps.register_curve_network(name, points, edges, **kwargs)
   
    def curvenetwork(self, name, nodes, edges, **kwargs):
        radius = kwargs.get('radius', self.radius)
        return ps.register_curve_network(name, nodes, edges, radius=radius)

    def pointcloud(self, name, pointcloud, color=None, radius=None, **kwargs):
        """Visualize non-zero entries of heat map on 3D point cloud.
            point cloud (torch.Tensor, [N, 3])
        """
        if radius is None:
            radius = self.radius
        if color is None:
            return ps.register_point_cloud(name, pointcloud, radius=radius, **kwargs)
        else:
            return ps.register_point_cloud(
                name, pointcloud, radius=radius, color=color, **kwargs
                )
    
    def get_meshes(self, centers, eigvals, eigvecs):
        """ Prepare corners and faces (for visualization only). """

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
        corners, faces = self.get_meshes(planes[:, :3], planes[:, 6:8], planes[:, 8:14])
        return ps.register_surface_mesh(name, corners, faces)

    def boxes_from_attr(self, name, attr, labels=None, **kwargs):
        from pcdet.utils.box_utils import boxes_to_corners_3d
        corners = boxes_to_corners_3d(attr)
        if 'with_ori' in kwargs:
            with_ori = kwargs.pop('with_ori')
        else:
            with_ori = False
        ps_box = self.boxes(name, corners, labels, **kwargs)
        if with_ori:
            ori = attr[:, -1]
            sint, cost = np.sin(ori), np.cos(ori)
            arrow = np.stack([sint, cost, np.zeros_like(cost)], axis=-1)[:, np.newaxis, :].repeat(8, 1)
            ps_box.add_vector_quantity('orientation', arrow.reshape(-1, 3), enabled=True)
        

    def boxes(self, name, corners, labels=None, **kwargs):
        """
            corners (shape=[N, 8, 3]):
            labels (shape=[N])
        """
        edges = [[0, 1], [0, 3], [0, 4], [1, 2],
                 [1, 5], [2, 3], [2, 6], [3, 7],
                 [4, 5], [4, 7], [5, 6], [6, 7]]
        N = corners.shape[0]
        edges = np.array(edges) # [12, 2]
        edges = np.repeat(edges[np.newaxis, ...], N, axis=0) # [N, 12, 2]
        offset = np.arange(N)[..., np.newaxis, np.newaxis]*8 # [N, 1, 1]
        edges = edges + offset
        if kwargs.get('radius', None) is None:
            kwargs['radius'] = 2e-4
        ps_box = ps.register_curve_network(
                     name, corners.reshape(-1, 3),
                     edges.reshape(-1, 2), **kwargs
                 )
        if kwargs.get('color', None) is None:
            if labels is not None:
                # R->Car, G->Ped, B->Cyc
                colors = np.array([[1,0,0], [0,1,0], [0,0,1], [0,1,1], [1,0,1], [1,1,0]])
                labels = np.repeat(labels[:, np.newaxis], 8, axis=-1).reshape(-1)
                ps_box.add_color_quantity('class', colors[labels],
                                          defined_on='nodes', enabled=True)

        return ps_box

    def wireframe(self, name, heatmap):
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
        ps.set_up_dir('z_up')
        ps.init()
        ps.show()

    def look_at(self, center, distance=100, bev=True, **kwargs):
        if bev:
            camera_loc = center + np.array([0, 0, distance])
            # look down from bird eye view
            # with +y-axis being the up dir on the image
            ps.look_at_dir(camera_loc, center, (0,1,0), **kwargs)
        else:
            raise ValueError("Not Implemented Yet, please use bev=True")

    def screenshot(self, filename, **kwargs):
        ps.screenshot(filename, **kwargs)

if __name__ == '__main__':
    vis = Visualizer([], [])
    vis.save('temp.pth')
