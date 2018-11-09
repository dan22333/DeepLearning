# Simple neural network for facial feature recognition [question 3.3]

import torch


class FaceNetP3(torch.nn.Module):
    def __init__(self):
        super(FaceNetP3, self).__init__()

        # linear part
        self.linear_part = torch.nn.Sequential (
            torch.nn.Linear(9216,100), #1
            torch.nn.ReLU(), #2
            torch.nn.Linear(100,30) #3
        )



    def forward(self, x):
        # reshape to FC layers
        x = x.view([-1, 9216])

        output = self.linear_part(x)
        return output

