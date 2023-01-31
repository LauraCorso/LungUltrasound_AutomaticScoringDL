# MEDICAL IMAGING DIAGNOSTIC PROJECT - Optimizer.py
# CORSO LAURA 230485

"""
This file contains the code for the definition of the optimizer.
Adam is used to update the parameters of the network during the training phase.
The optimizer is definied by assigning distinct learning rates to the parameters of the CNN and to the ones of the rest of the network.
This is done because the ResNet18 is pretrained, hece only a finetuning of its parameters is needed. Instead, the learning rate for
the rest of the network (Attention Layer + Bi-LSTM) is 10 times higher because its trained from scratch.
"""

"""
--------------------------- Imports ---------------------------
"""
from Config import *


"""
--------------------------- Adam optimizer ---------------------------
"""
"""
Function that defines the Adam optimizer used to update the parameters:
It is defined in order to: 
* finetune the pretrained CNN (ResNet18)
* train the rest of the network (Attention Layer + Bi-LSTM) from scratch.

:param model: instance of the network to be trained.
:param lr: learning rate coefficient. Default = LEARNING_RATE.
:param wd: weight decay coefficient. Default = WEIGHT_DECAY.
:return optimizer: SGD optimizer.
"""
def get_optimizer(model, lr = LEARNING_RATE, wd = WEIGHT_DECAY):
  # Creation of two groups of parameters: one for the CNN
  # and the other for rest of the layers of the network
  CNN_params = []
  rest_of_the_net_params = []
  
  # Iterate through the layers of the network
  for name, param in model.named_parameters():
    # The name of the parameters belonging to resnet18 starts with 0, while 
    # the ones belonging to G start with 1
    if name.startswith('resnet'):
        CNN_params.append(param)
    else:
        rest_of_the_net_params.append(param)
  
  # Assign the distinct learning rates to each group of parameters
  optimizer = torch.optim.Adam([{'params': CNN_params}, {'params': rest_of_the_net_params, 'lr': lr}], lr = (lr / 10), weight_decay = wd)
  
  return optimizer



"""
--------------------------- Testing of the code above ---------------------------
"""
if __name__ == "__main__":
  from Model import *

  net = Network()
  get_optimizer(net, LEARNING_RATE, WEIGHT_DECAY)