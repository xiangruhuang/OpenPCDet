import plotly
import plotly.graph_objs as go
import torch
from torch import nn

class PlotlyVisualizer(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()

    def forward(self, path):
        batch_dict = torch.load(path, map_location='cpu')
        point_bxyz = batch_dict['point_bxyz']
        plotly.offline.init_notebook_mode()
        for b in range(batch_dict['batch_size']):
            mask = point_bxyz[:, 0] == b
            point_xyz = point_bxyz[mask, 1:4]
            trace = go.Scatter3d(
                x=point_xyz[:, 0],
                y=point_xyz[:, 1],
                z=point_xyz[:, 2],
                mode='markers',
                marker={
                    'size': 1,
                    'opacity': 0.8,
                }
            )
            layout = go.Layout(
                margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
                scene=dict(
                    xaxis=dict(title="x", range = [-100,100]),
                    yaxis=dict(title="y", range = [-100,100]),
                    zaxis=dict(title="z", range = [-100,100])
                )
            )
            data = [trace]
            plot_figure = go.Figure(data=data, layout=layout)
            plotly.offline.iplot(plot_figure)