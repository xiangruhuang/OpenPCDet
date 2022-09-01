import numpy as np
import open3d as o3d
import os, os.path as osp
from pcdet.datasets.surreal.smpl_utils import SMPLModel

def laplacian(mesh):
  """Returns the laplacian matrix of this mesh.

  """
  faces = np.array(mesh.triangles)
  N = np.array(mesh.vertices).shape[0]
  A = np.zeros((N, N))
  for i in range(3):
    for j in range(3):
      if i == j:
        continue
      A[(faces[:, i], faces[:, j])] = 1.0
  A = A + A.T
  diag = A.dot(np.ones(N))
  L = np.diag(diag) - A
  return L

def laplacian_embedding(mesh, rank=30):
  faces = np.array(mesh.triangles)
  N = np.array(mesh.vertices).shape[0]
  A = np.zeros((N, N))
  for i in range(3):
    for j in range(3):
      if i == j:
        continue
      A[(faces[:, i], faces[:, j])] = 1.0
  A = A + A.T
  diag = A.dot(np.ones(N))
  L = np.diag(diag) - A
  eigvals, eigvecs = np.linalg.eigh(L)
  embedding = eigvecs[:, 1:(rank+1)]
  return embedding

def floyd(mesh):
  faces = np.array(mesh.triangles)
  N = np.array(mesh.vertices).shape[0]
  Dist = np.zeros((N, N)) + 1e10
  for i in range(N):
    Dist[i, i] = 0.0
  for i in range(3):
    for j in range(3):
      if i == j:
        continue
      Dist[(faces[:, i], faces[:, j])] = 1.0
  #for k in range(N):
  #  print(k, N)
  #  for i in range(N):
  #    for j in range(N):
  #      if (i == j) or (i == k) or (j == k):
  #        continue
  #      if Dist[i, j] > Dist[i, k] + Dist[k, j]:
  #        Dist[i, j] = Dist[i, k] + Dist[k, j]
  return Dist

if __name__ == '__main__':
  smpl_model = SMPLModel('data/surreal/smpl_male_model.mat')
  smpl_model.update_params(np.zeros(85))

  mesh = o3d.geometry.TriangleMesh()
  import argparse
  parser = argparse.ArgumentParser('Compute Laplacian embedding of a given mesh (.ply)')
  parser.add_argument('--mesh', default=None, type=str,
    help='input mesh (.ply)')
  parser.add_argument('--mat', default=None, type=str,
    help='output dictionary (.mat)')
  parser.add_argument('--name', default=None, type=str,
    help='name')
  args = parser.parse_args()
 
  mesh.vertices = o3d.utility.Vector3dVector(smpl_model.verts)
  mesh.triangles = o3d.utility.Vector3iVector(smpl_model.faces)
 
  embedding = laplacian_embedding(mesh, rank=128)
  embedding = (embedding - embedding.min(0)[np.newaxis, :])/(embedding.max(0)-embedding.min(0))[np.newaxis, :]
  np.save(args.mat, embedding)
