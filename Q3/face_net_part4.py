# Convolutional neural network for facial feature recognition [question 3.4]


import torch


class FaceNetP4(torch.nn.Module):
    def __init__(self):
        super(FaceNetP4, self).__init__()

        # convolutional part
        self.conv_part = torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, 3),              # 1
        torch.nn.MaxPool2d(2, stride=2),        # 2
        torch.nn.ReLU(),                        # 3
        torch.nn.Conv2d(32, 64, 2),             # 4
        torch.nn.MaxPool2d(2, stride=2),        # 5
        torch.nn.ReLU(),                        # 6
        torch.nn.Conv2d(64, 128, 2),            # 7
        torch.nn.MaxPool2d(2, stride=2),        # 8
        torch.nn.ReLU(),                        # 9
        )

        # linear part (fully connected layers)
        self.linear_part = torch.nn.Sequential(
            torch.nn.Linear(15488, 500),        # 11
            torch.nn.ReLU(),                    # 12
            torch.nn.Linear(500, 500),          # 13
            torch.nn.ReLU(),                    # 14
            torch.nn.Linear(500, 30),
        )

    def forward(self, x):
        x = self.conv_part(x)

        # reshape to FC layers
        x = x.view([-1, 15488])

        output = self.linear_part(x)
        return output

