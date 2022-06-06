from torch import nn

def MLP(channels, activation=nn.LeakyReLU(0.2), bn_momentum=0.1, bias=True):
    return nn.Sequential(
        *[
            nn.Sequential(
                nn.Linear(channels[i - 1], channels[i], bias=bias),
                nn.BatchNorm1d(channels[i], momentum=bn_momentum),
                activation,
            )
            for i in range(1, len(channels))
        ]
    )
