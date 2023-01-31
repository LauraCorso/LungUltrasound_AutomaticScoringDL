# MEDICAL IMAGING DIAGNOSTIC PROJECT - CostFunction.py
# CORSO LAURA 230485

"""
This file contains the code for the definition of the SORD loss.
For its definition check the paper at https://ieeexplore.ieee.org/document/9093068 
"""


"""
--------------------------- Imports ---------------------------
"""
from Config import *


"""
--------------------------- Loss definition ---------------------------
"""
"""
Function that return the value of the SORD loss between the predictions of the network and the ground truth labels.

:param preds: tensor of size B x NC (B = number of videos in a batch, NC = number of classes) 
              that contains the predictions of the network.
:param ground_truth: tensor of size B (number of videos in a batch) 
                     containing the ground truth labels.
:param num_classes: integer number of classification classes. Default = 4.
:param multiplier: integer constant factor to which the distance function between scores is multiplied. Default = 2.
:param wide_gap_loss: boolean that controls whether to increase the distance of score 0 
                      from the others (True) or not (False). Default = False.
:return loss: tensor containing the SORD loss between preds and ground_truth.
"""
def sord_loss(preds, ground_truth, num_classes = CLASSES, multiplier = 2, wide_gap_loss = False):
    # Allocation of the SORD probability vector.
    # It is a vector of size [B, NC] where B is the number of videos contained in a batch and NC is the number of classes.
    # It contains the ground truth informations encoded as soft-valued vectors.
    # Hence, for a video V with score S the (V, I) element of the vector is computed as exp(-d(S, I))/sum_over_J_from_NC_values(exp(-d(J, I)))
    # where d() is the distance function between scores.
    batch_size = ground_truth.shape[0]
    labels_sord = np.zeros((batch_size, num_classes), dtype = 'float32')

    # Filling of the SORD probability vector
    for video_idx in range(batch_size):
        current_label = ground_truth[video_idx].item()
    
        # For each class compute the distance to the ground truth label
        for class_idx in range(num_classes):
            # Computation that increases the distance of score 0 from the others
            if wide_gap_loss:
                wide_label = current_label
                wide_class_idx = class_idx
                # Increases the gap between positive and negative patients
                if wide_label == 0:
                    wide_label = -0.5
                if wide_class_idx == 0:
                    wide_class_idx = -0.5

                labels_sord[video_idx][class_idx] = multiplier * abs(wide_label - wide_class_idx) ** 2

            # Standard computation distance = 2 * ((class label - ground truth label))^2
            else:
                labels_sord[video_idx][class_idx] = multiplier * abs(current_label - class_idx) ** 2

        

    print("labels_sord: ", labels_sord)
    labels_sord = torch.from_numpy(labels_sord).to(T_DEVICE) #conversion in torch Tensor
    labels_sord = F.softmax(-labels_sord, dim=1)

    # Rescaling the predictions so that the elements lie in the range [0,1] and sum to 1
    print("Preds: ", preds)
    log_predictions = F.log_softmax(preds, 1)
    print("Log_predictions: ", log_predictions)

    # Computation of the Cross-Entropy Loss between ground truth labels and predictions
    loss = (-labels_sord * log_predictions).sum(dim = 1).mean()
    print("Loss: ", loss)

    return loss



"""
--------------------------- Testing of the code above ---------------------------
"""
if __name__ == "__main__":
  from Data import *
  from Model import *

  # Instantiation of the train and test dataloaders
  ta, te = get_data(BATCH_SIZE, ds_path)

  # Instantiation of the network
  net = Network()
  net.to(T_DEVICE)

  for idx, (v, l, p) in enumerate(te):
    if idx == 0:
      print("Input: ", v.shape)
      print("Input type: ", type(input))
      l.to(T_DEVICE)
      output = net(v.to(T_DEVICE))
      print("Output size: ", output.shape)
      print("Output: ", output)
      print("Output type: ", type(output))
      loss = sord_loss(output, l)
      print("Loss: ", loss.item())
      print(p)
    else:
      break