# MEDICAL IMAGING DIAGNOSTIC PROJECT - Main.py
# CORSO LAURA 230485

"""
This project is made up of the following files:

    - Main.py that contains the code for trainig, testing and evaluation of the results.
    - Config.py that contains the imports and the global variables shared among the files.
    - GDriveManagement.py that contains the code that is used to prepare the data of the dataset Dati San Matteo 2 in a compressed format in order to use them on the Azure machine.
    - Dataset.py that handles the data needed for classification.
    - LenghtAnalysis.py that contains the code to analyze what is the best approach to get videos all of the same size in a batch.
    - Model.py that contains the code for the definition of the structure of the network.
    - CostFunction.py that contains the code for the definition of the SORD loss.
    - Optimizer.py that contains the code for the Stochastic Gradient Descent optimizer.
"""


"""
--------------------------- Imports ---------------------------
"""
from Config import *
from Data import *
from Model2 import *
from Optimizer import *
from CostFunction import *


"""
--------------------------- Functions for NaN debugging ---------------------------
"""

"""
Function for tensor hook
"""
def hook_t(grad):
  print("Grad: ", grad)

"""
Function for non convolutional layer hook
"""
def hook_m(self, inp, out):
  print("Inside: ", self.__class__.__name__)
  print("grad_input: ", inp[0].isnan().any())
  print("grad_output: ", out[0].isnan().any())

"""
Function for convolutional layer hook
"""
def hook_res(self, inp, out, name = None):
  print(name)
  for grad in inp:
    if grad is not None:
      print("   grad: ", grad[0].isnan().any())
    else:
      print("   grad: ", 'None')

  if out is not None:
    print("   grad: ", out[0].isnan().any())
  else:
    print("   grad: ", 'None')

"""
Function to plot the gradients flowing through different layers in the net during training.
Can be used for checking for possible gradient vanishing / exploding problems.

Usage: Plug this function in Trainer class after loss.backwards() as 
       "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow
"""
def plot_grad_flow(named_parameters):
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    fig = plt.figure(1, figsize = (20, 15))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c") #in cyan are visualized the maximum value of the gradient for the given parameter
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b") #in blue are visualized the average value of the gradient for the given parameter
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) #zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    t = "Gradients" + ".png"
    fig.savefig(os.path.join(result_path, t))

"""
--------------------------- Training step ---------------------------
This section describes the pipeline adopted for the supervised training of the model.
"""
"""
Function that defines the training procedure.

:param net: network to train.
:param data_loader: dataloader for the training.
:param optimizer: optimizer used to update the parameters of the network.
:param cost_function: cost function for the loss computation.
:param device: GPU where the network is trained.
:return: training loss, training accuracy.
"""
def training_step(net, data_loader, optimizer, cost_function, device = T_DEVICE):
#def training_step(net, optimizer, cost_function, device = T_DEVICE):
  print("\nTRAINING STEP")
  samples = 0. #integer that contains the total number of videos
  cumulative_loss = 0. #float that contains the sum of the losses through the batches
  cumulative_accuracy = 0. #float that contains the sum of the accuracies through the batches

  # Set the network to training mode
  net.train() 

  """# Register the backward hook for every convolutional layer of the model
  for m in net.named_modules():
    if isinstance(m[1], torch.nn.Conv2d):
      m[1].register_full_backward_hook(hook = partial(hook_res, name = m))"""

  """# Register the backward hook for every module of the network
  net.fc.register_full_backward_hook(hook_m)
  net.att.register_full_backward_hook(hook_m)
  net.bilstm.register_full_backward_hook(hook_m)
  net.resnet.register_full_backward_hook(hook_res)"""

  # Iterate over the training set
  for batch_idx, (videos, labels, paths) in enumerate(data_loader):
    print("\nBatch n. ", batch_idx)
    # Load data into GPU
    videos = videos.to(device)
    labels = labels.to(device)

    # Gradients reset
    optimizer.zero_grad()

    """for video in videos:
      print("Video contains NaN values: ", video.isnan().any())"""
      
    # Forward pass
    outputs = net(videos)
    print("Outputs: ", outputs)

    # Loss computation
    loss = cost_function(outputs, labels)

    # Backward pass
    loss.backward()

    """# Instantiatethe graph of the flow of the gradient
    plot_grad_flow(net.named_parameters())"""

    #nn.utils.clip_grad_norm_(net.parameters(), 1)
    
    # Parameters update
    optimizer.step()
    
    # Gradients reset
    optimizer.zero_grad()

    # Fetch prediction and loss value
    samples += videos.shape[0]
    cumulative_loss += loss.item()
    _, predicted = outputs.max(dim=1) #extract the index of the most probable label, i.e. extract the prediction as a label (0 - 4)

    # Compute training accuracy
    cumulative_accuracy += predicted.eq(labels).sum().item()

  
  """
  Section that reproduces the error with random input.
  """
  """
  videos = torch.rand(4, 252, 3, 256, 256)
  labels = torch.from_numpy(np.array([0, 3, 2, 1]))
  # Load data into GPU
  videos = videos.to(device)
  labels = labels.to(device)

  for video in videos:
    print(video.isnan().any())
    
  # Forward pass
  outputs = net(videos)
  print("Outputs: ", outputs)
  outputs.register_hook(hook_t)

  # Loss computation
  loss = cost_function(outputs, labels)
  print("loss: ", loss)
  loss.register_hook(hook_t)


  # Backward pass
  loss.backward()

  #nn.utils.clip_grad_norm_(net.parameters(), 1)

  for name, param in net.named_parameters():
    print(name, torch.isfinite(param.grad).all())
  
  # Parameters update
  optimizer.step()
  
  # Gradients reset
  optimizer.zero_grad()

  # Fetch prediction and loss value
  samples += videos.shape[0]
  cumulative_loss += loss.item()
  _, predicted = outputs.max(dim=1) #extract the index of the most probable label, i.e. extract the prediction as a label (0 - 4)

  # Compute training accuracy
  cumulative_accuracy += predicted.eq(labels).sum().item()
  """

  return cumulative_loss/samples, cumulative_accuracy/samples*100


"""
--------------------------- Testing step ---------------------------
This section describes the pipeline adopted to test a model previously trained.
The network is evaluated through F1 Score, Precision and Recall.
"""
"""
Definition of the test procedure

:param net: network to test.
:param data_loader: dataloader for the testing.
:param cost_function: cost function for the loss computation.
:param device: GPU where the network is trained.
:return: test loss, test accuracy, F1 Score, Precision, Recall.
"""

def test_step(net, data_loader, cost_function, device = T_DEVICE):
  print("\nTEST STEP")
  samples = 0. #integer that contains the total number of videos
  cumulative_loss = 0. #float that contains the sum of the losses through the batches
  cumulative_accuracy = 0. #float that contains the sum of the accuracies through the batches
  preds = [] #list that will contain the labels predicted by the network for each video in the test set
  trues = [] #list that will contain the ground truth labels for each video in the test set

  # Set the network to evaluation mode
  net.eval() 

  # Disable gradient computation 
  with torch.no_grad():
    # Iterate over the test set
    for batch_idx, (videos, labels, paths) in enumerate(data_loader):
      print("\nBatch n. ", batch_idx)
      # Load data into GPU
      videos = videos.to(device)
      labels = labels.to(device)
        
      # Forward pass
      outputs = net(videos)

      # Loss computation
      loss = cost_function(outputs, labels)
      print("Loss: ", loss)

      # Fetch prediction and loss value
      samples = samples + videos.shape[0]
      print("Samples: ", samples)
      cumulative_loss = cumulative_loss + loss.item() 
      print("C Loss: ", cumulative_loss)
      _, predicted = outputs.max(1)
      preds.append(predicted)
      trues.append(labels)

      # Compute accuracy
      cumulative_accuracy = cumulative_accuracy +  predicted.eq(labels).sum().item()

  # Computation of precision, recall and F1 score globally on the whole test set
  precision, recall, fscore, _ = precision_recall_fscore_support(y_true = torch.cat(trues).cpu(), y_pred = torch.cat(preds).cpu(), average = 'micro')

  return cumulative_loss/samples, cumulative_accuracy/samples*100, fscore, precision, recall


"""
--------------------------- Training of the model ---------------------------
This section train the model.
"""
#torch.autograd.set_detect_anomaly(True)
# Creation of the resul folder
if not os.path.exists(result_path):
    os.mkdir(result_path)

# Instantiation of the train and test dataloaders
train_dl, test_dl = get_data()

# Instantiation of the model
print("\nInstantiation of the model")
net = Network()
net.to(T_DEVICE)

# Instantiation of the optimizer
optimizer = get_optimizer(net)

# Instantiation of the cost function
cost_function = sord_loss

# Variables that will contain the best model found
best_net = net
best_F1 = 0.0     #(poor) 0 <= F1 <= 1 (optimum)
best_precision = 0.0    #(poor) 0 <= precision <= 1 (optimum)
best_recall = 0.0     #(poor) 0 <= recall <= 1 (optimum)

# Range over epochs to train the model
tr_accuracies = [] # List that stores the training accuracies over the epochs
tr_losses = [] # List that stores the training losses over the epochs
te_accuracies = [] # List that stores the test accuracies over the epochs
te_losses = [] # List that stores the test losses over the epochs
f1s = [] # List that stores the F1 scores over the epochs
precisions = [] # List that stores the precisions over the epochs
recalls = [] # List that stores the recalls over the epochs
for e in range(EPOCHS):
  training_step(net, train_dl, optimizer, cost_function)
  train_loss, train_accuracy, _, _, _ = test_step(net, train_dl, cost_function)
  test_loss, test_accuracy, fscore, precision, recall = test_step(net, test_dl, cost_function)

  tr_accuracies.append(train_accuracy)
  tr_losses.append(train_loss)
  te_accuracies.append(test_accuracy)
  te_losses.append(test_loss)
  f1s.append(fscore)
  precisions.append(precision)
  recalls.append(recall)

  print('Epoch: {:d}'.format(e+1))
  print('\t Training loss {:.5f}, Training accuracy {:.2f}'.format(train_loss, train_accuracy))
  print('\t Test loss {:.5f}, Test accuracy {:.2f}'.format(test_loss, test_accuracy))
  print('-----------------------------------------------------')

  if fscore > best_F1:
    best_F1 = fscore
    best_net = net
    best_precision = precision
    best_recall = recall

# Saving the results
with open(os.path.join(result_path, "Training_acc.txt"), 'w') as f:
  for item in tr_accuracies:
    f.write("%s\n" % item)

with open(os.path.join(result_path, "Training_l.txt"), 'w') as f:
  for item in tr_losses:
    f.write("%s\n" % item)

with open(os.path.join(result_path, "Testin_acc.txt"), 'w') as f:
  for item in te_accuracies:
    f.write("%s\n" % item)

with open(os.path.join(result_path, "Testing_l.txt"), 'w') as f:
  for item in te_losses:
    f.write("%s\n" % item)

with open(os.path.join(result_path, "F1Scores.txt"), 'w') as f:
  for item in f1s:
    f.write("%s\n" % item)

with open(os.path.join(result_path, "Precisions.txt"), 'w') as f:
  for item in precisions:
    f.write("%s\n" % item)

with open(os.path.join(result_path, "Recalls.txt"), 'w') as f:
  for item in recalls:
    f.write("%s\n" % item)

torch.save(best_net, result_path)