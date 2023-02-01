# MEDICAL IMAGING DIAGNOSTIC PROJECT - Model.py
# CORSO LAURA 230485

"""
This file contains the code for the definition of the network's model composed of:
* a pretrained ResNet18 
* an attention layer composed of three convolutional layer (for its definition see the paper at https://www.mdpi.com/2076-3417/11/24/11697 )
* a bidirectional LSTM
* a fully connected layer.
"""


"""
--------------------------- Imports ---------------------------
"""
from Config import *
from Data import *


"""
--------------------------- Structure of the model ---------------------------
"""
"""
Class that defines the three layers and the computation performed by the Attention Layer.
"""
class AttentionLayer(nn.Module):

  """
  Function runned once when instantiating the AttentionLayer object.
  It defines the layers of the network.
  """
  def __init__(self):
    super(AttentionLayer, self).__init__()

    self.conv1 = nn.Conv2d(in_channels = 512, out_channels = 16, kernel_size = 3, padding = 1)
    self.bn1 = nn.BatchNorm2d(16) #InstanceNorm2d calculate the mean and standard-deviation per-dimension separately for each object in a mini-batch. It is used here because we evaluate one frame at a time --> batches will contain only one frame
    self.relu1 = nn.ReLU()

    self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = 3, padding = 1)
    self.bn2 = nn.BatchNorm2d(8) #InstanceNorm2d calculate the mean and standard-deviation per-dimension separately for each object in a mini-batch. It is used here because we evaluate one frame at a time --> batches will contain only one frame
    self.relu2 = nn.ReLU()

    self.conv3 = nn.Conv2d(in_channels = 8, out_channels = 1, kernel_size = 3, padding = 1)
    self.sig3 = nn.Sigmoid()

  """
  Function that defines the computation performed by the network at every call.

  :param x: tensor input of the network.
  :return x: tensor output of the computation.
  """
  def forward(self, x):
    # First convolutional layer + Batch Normalization + ReLu
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu1(x)

    # Second convolutional layer + Batch Normalization + ReLu
    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu2(x)

    # Third convolution layer + Sigmoid
    x = self.conv3(x)
    x = self.sig3(x)

    return x



"""
Class that defines a network composed by 
- a pretrained ResNet18 
- an attention layer composed of three convolutional layer
- a bidirectional LSTM
- a fully connected layer.
"""
class Network(nn.Module):

  """
  Function runned once when instantiating the Network object.
  It defines the layers of the network.
  """
  def __init__(self):
    super(Network, self).__init__()

    # Load the pre-trained ResNet18
    cnn = torchvision.models.resnet18(weights = torchvision.models.ResNet18_Weights.DEFAULT)
    # Remove the last fully connected layer to get the output feature map
    self.resnet = nn.Sequential(*(list(cnn.children())[:-1]))

    # Define the attention layer
    self.att = AttentionLayer()

    # Define the bidirectional LSTM
    self.bilstm = nn.LSTM(512, HS, batch_first = True, bidirectional = True)

    # Define the fully connected layer
    self.fc = nn.Linear(252*HS*2, CLASSES)

  """
  Function that defines the computation performed by the network at every call.

  :param i: tensor input of the network. It is a matrix [B x L x C x W x H] where
                                         B is the number of videos in a batch
                                         L is the lenght of each video in the batch
                                         C is the number of channel of each frame in the video
                                         W is the width of each frame in the video
                                         H is the height of each frame in the video.
            Therefore, x contains B videos of L frames and each frame is an image of size C x W x H.
  :return x: tensor output of the computation. It is a matrix [B x NC] (NC = number of classes) 
             that contains, for each class, the probability that each video of the batch is of that class.
  """
  def forward(self, i):
    # The input tensor in the test step in unbatched --> add the B dimension
    if len(list(i.shape)) < 5:
      i = torch.unsqueeze(i, dim = 0)

    # Reshape the input tensor from [B x L x C x W x H] to [B*L x C x W x H] to input data in the CNN
    x = i.reshape(-1, i.shape[2], i.shape[3], i.shape[4])
    print("Input NEt: ", x.shape)

    # Compute the spatial feature map (output of the CNN)
    sfm = self.resnet(x)
    print("sfm: ", sfm.shape)

    # Compute the the attention map
    mask = self.att(sfm)

    # Extract the number of channel of the feature map
    [_, c, _, _] = sfm.shape

    # Expand the mask in the channel dimension.
    # The mask is a matrix [B*L x 1 x H x W]. The channel dimension is increased to match the number of channel of the feature
    # map and do the element-wise multiplication
    mask = mask.expand(mask.shape[0], c, mask.shape[1], mask.shape[2])

    # Element-wise multiplication between the feature map and the attention map
    x = torch.mul(sfm, mask)

    # Global Average Pooling
    gap = nn.AvgPool2d(x.size()[2:])
    x = gap(x)
    print("After AL: ", x.shape)

    # Reshape the tensor from [B*L x C x W x H] to [B, L, C*W*H] to obtain the feature vector for each frame of the videos in a batch
    x = x.reshape(i.shape[0], i.shape[1], -1)

    # Compute the temporal feature map (output of the RNN)
    x, _ = self.bilstm(x)
    print("After LSTM: ", x.shape)

    # Reshape the tensor from [B, L, HS] to [B, L*HS] where HS is the number of features in the hidden state of the Bi-LSTM
    x = x.reshape(x.shape[0], -1)

    # Compute the output of the net
    x = self.fc(x)

    return x



"""
--------------------------- Testing of the code above ---------------------------
"""
if __name__ == "__main__":
  from Data import *

  ta, te = get_data(BATCH_SIZE, ds_path)
  net = Network()
  #net.eval()

  for idx, (v, l, p) in enumerate(te):
    if idx == 0:
      print("Input: ", v.shape)
      output = net(v)
      print("Output size: ", output.shape)
      print("Output: ", output)
    else:
      break