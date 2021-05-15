import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNCUDA(nn.Module):
    def __init__(self, in_channels=5, n_actions=14):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels (1 more for goals)
            n_actions (int): number of outputs
        """
        super(DQNCUDA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, input):
        x, a, g = input
        x = x.transpose(1, 0, 2, 3)
        g = g.transpose(3, 0, 1, 2)
        x = torch.from_numpy(x).to("cuda")
        a = torch.from_numpy(a).to("cuda")
        g = torch.from_numpy(g).to("cuda")
        concat = torch.cat([x, g])
        concat = concat.permute(1, 0, 2, 3)
        normalized = concat.float() / 255
        x = F.relu(self.conv1(normalized))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        output = self.head(x)
        filtered_output = torch.multiply(output, a)
        return filtered_output


class VanillaDQNCUDA(nn.Module):
    def __init__(self, in_channels=4, n_actions=14):
        """
        Initialize Deep Q Network
        Args:
            in_channels (int): number of input channels
            n_actions (int): number of outputs
        """
        super(VanillaDQNCUDA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        x = torch.from_numpy(x).to("cuda")
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)