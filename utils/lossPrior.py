import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


class LossPrior(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(LossPrior, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        pred_target1_squeeze = torch.squeeze(pred_target1)
        #print(type(pred_target1_squeeze))  <class 'torch.Tensor'>
        #print(pred_target1.size())  [1,19,512,1024]
        #print(pred_target1_squeeze.size())  [1,19,512,1024]
        pred_target1_squeeze = pred_target1_squeeze.cpu().detach()
        #print(type(pred_target1_squeeze))   <class 'torch.Tensor'>

        pred_target1_squeeze_shape = pred_target1_squeeze.view(19,512*1024)
        #print(pred_target1_squeeze_shape.size())  [19, 524288]
        #print(type(pred_target1_squeeze_shape))   <class 'torch.Tensor'>


        pred_target1_squeeze_shape = pred_target1_squeeze_shape.numpy()
        #print(type(pred_target1_squeeze_shape))  <class 'numpy.ndarray'>


        pred_target1_squeeze_shape = pred_target1_squeeze_shape.sum(1)
        #print(type(pred_target1_squeeze_shape))   <class 'numpy.ndarray'>

        target_fenbu = pred_target1_squeeze_shape/(512*1024)
        #print(type(target_fenbu))  <class 'numpy.ndarray'>

        target_fenbu = torch.from_numpy(target_fenbu).float()
        #print(type(target_fenbu))   <class 'torch.Tensor'>



        #print(pred_target1_squeeze_shape)   [19,1]


        source_fenbu=Variable(torch.Tensor(np.array([0.14250927,0.08276533,0.11154246,0.02590515,0.02759069,0.05180943,0.02695177,0.02489249,0.12235474,0.03795863,0.09590349,0.04986444,0.01680536,0.06009313,0.02781029,0.02729403,0.01970997,0.02439774,0.02384157])))
        #print(type(source_fenbu))  <class 'torch.Tensor'>

        #source_fenbu=Variable(source_fenbu,requires_grad=False)
        #source_fenbu = np.array(source_fenbu)
        source_fnebu = source_fenbu / 2

        #target_fenbu = Variable(target_fenbu,requires_grad=True)

        distance = source_fenbu - target_fenbu
        for i in range(19):
            if(distance[i] < 0):
                distance[i] = 0
        #print(distance)

        #print(type(distance))  <class 'torch.Tensor'>
        loss_prior = distance.sum()


        #loss_prior = torch.from_numpy(np.array(loss_prior))
        #loss_prior = Variable(loss_prior,requires_grad=True)


        loss_prior = loss_prior.to(device)
        return loss_prior
