from utils import *


class TripletLoss(nn.Module):
    def __init__(self, distance_net, margin):
        super(TripletLoss, self).__init__()
        self.distance_net = distance_net
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = self.distance_net(anchor, positive)
        neg_dist = self.distance_net(anchor, negative)
        return F.relu(pos_dist - neg_dist + self.margin).mean()

