import numpy as np
import polyscope as ps
import sys

ps.init(); ps.set_up_dir('z_up')
f = sys.argv[1]
pcd = np.load(f)
seg = np.load(f.replace('.npy', '_seg.npy'))[:, 1]
pcd = pcd[:seg.shape[0], :3]
print(seg.shape, pcd.shape)
ps_p = ps.register_point_cloud('points', pcd, radius=2e-4)
ps_p.add_scalar_quantity('seg', seg)

ps.show()
