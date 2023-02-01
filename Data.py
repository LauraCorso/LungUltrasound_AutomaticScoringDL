# MEDICAL IMAGING DIAGNOSTIC PROJECT - Data.py
# CORSO LAURA 230485

"""
Code that handles the data needed for classification. It is composed of:
* code to extract the data from the compressed version of the dataset
* code to define a custom Dataset class 
* code to instantiate the Dataloaders for training and testing.
"""


"""
--------------------------- Imports ---------------------------
"""
from Config import *


"""
--------------------------- Load of the data ---------------------------
"""
print("\n\nLoad of the data...")
compressed_dataset = os.path.join(root_path, "DatasetC")

# Creation of the folder for the uncompressed dataset
if not os.path.exists(ds_path):
    print("\nUnzipping the dataset...")
    os.mkdir(ds_path)

    for name in os.listdir(compressed_dataset):
        #select only the folders containing the .mat files
        if name.startswith("1"):
            #create the folder for the patient
            if not os.path.exists(os.path.join(ds_path,  name[:-4])):
                os.mkdir(os.path.join(ds_path,  name[:-4]))

            #extract the videos 
            with ZipFile(os.path.join(compressed_dataset, name), 'r') as zipObj:
                print("Extracting ", name[:-4], "...")
                zipObj.extractall(os.path.join(ds_path,  name[:-4]))
                print("     Extracted\n")
                
        elif name.startswith("A"):
            print("Copying ", name, "...")
            shutil.copy(os.path.join(compressed_dataset, name), os.path.join(ds_path,  name))
            print("     Copied\n")
else:
    print("Dataset already unzipped")


"""
--------------------------- Instatiation of the custom Dataset class ---------------------------

A custom Dataset class is needed in order to iterate through the dataset one video at a time.
"""
class VideoDataset(Dataset):

  """
  Function runned once when instantiating the Dataset object. 

  :param annotations_file: file.txt containing for each row the path of each video and its label in the format (path_to_video, label).
  :param video_dir: the directory containing the videos (string).
  :param video_transform: define possibile transformation that can be applied to the videos. Default: None.
  :param frame_transform: define possibile transformation that can be applied to the frames of the videos. Default: None.
  """
  def __init__(self, annotations_file, video_dir, video_transform = None, frame_transform = None):
    self.video_labels = pd.read_csv(annotations_file, sep = ";", header = None, names = ["path", "label"])
    self.video_dir = video_dir
    self.video_transform = video_transform
    self.frame_transform = frame_transform

  """
  Function that returns the number of videos in the dataset.

  :return: the integer number of videos in the dataset.
  """
  def __len__(self):
    return len(self.video_labels)

  """
  Function that loads and returns a video from the dataset at the given index.

  :param idx: integer index of the sample to return.
  :return video: tensor containing the video at index idx.
  :return label: tensor containing the label of the video at index idx.
  :return video_path: string containing the path to the video at index idx.
  """
  def __getitem__(self, idx):
    # Extraction of the path and of the label of the video at idx
    video_path = os.path.join(self.video_dir, self.video_labels.iloc[idx, 0])
    label = self.video_labels.iloc[idx, 1]

    # Construction of the video from the frames
    video_frames = [] 
    for frame_name in os.listdir(video_path):
      frame = Image.open(os.path.join(video_path, frame_name))
      video_frames.append(self.frame_transform(frame))

    # Transformation of the video in a Tensor
    video = torch.stack(video_frames, 0)

    # Application of the transofrmations, if present
    if self.video_transform:
      video = self.video_transform(video)

    return video, label, video_path


"""
--------------------------- Instatiation of the Dataloaders ---------------------------
"""
# Train/Test split

#list of the patients
patients = os.listdir(ds_path)
patients = [name for name in patients if name.startswith("1")]

#filter out patient n.1047 and n.1051
patients = list(filter(lambda i: i != '1047' and i != '1051', patients))

#split in train and test
p_train, p_test = train_test_split(patients, train_size = 0.8, random_state = 5)

#add patient n.1047 and n.1051 to the test set
p_test.append('1047')
p_test.append('1051')

#creation of two separated annotation files: one for the test and one for the training patients
test = open(os.path.join(ds_path, "Test_Annotations.txt"), 'w')
train = open(os.path.join(ds_path, "Train_Annotations.txt"), 'w')

#filling of the files
with open(os.path.join(ds_path, "Annotations.txt")) as f:
  for line in f.readlines():
    patient = line.split("/")[0] #extract the patient

    #assign the patient according to the split
    if patient in p_train:
      train.write(line)

    elif patient in p_test:
      test.write(line)

    else:
      print(line + " not present in the dataset")

f.close()
test.close()
train.close()



# Definition of a custom transformation in order to perform the zero-padding of a video in the time domain.
# This video level transformation is needed in oder to form batches with videos of the same length.
# Note: the zero padding is applied both in training and test set to use the same network with the last fc layer of the right dimension.
"""
Function that finds the maximum video length in the training dataset.

:return max_len: the integer maximum video length in the training dataset.
"""
def max_len():
  #prepare data transformations for the dataloader
  transform = list()
  transform.append(T.Resize((256, 256)))                      
  transform.append(T.ToTensor())                              
  transform.append(T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]))  
  transform = T.Compose(transform)  

  #instantiation of the train dataset
  t_d = VideoDataset(annotations_file = os.path.join(ds_path, "Train_Annotations.txt"), video_dir = os.path.join(root_path, "Dataset/"), video_transform = None, frame_transform = transform)

  max_len = 0
  for video, _, _ in t_d:
    if video.shape[0] > max_len:
      max_len = video.shape[0]

  return max_len


"""
Callable class that defines the zero-padding of a video in the time domain. 
"""
class ZeroPadding(object):

  """
  Function runned once when instantiating the ZeroPadding object. 

  :param output_size: integer that specify the final size of the video after zero-padding.
  """
  def __init__(self, output_size):
    assert isinstance(output_size, int) #output_size must be an integer
    self.output_size = output_size

  """
  Function runned at each instatiation of a ZeroPadding object.

  :param video: tensor containing the video to zero-pad.
  """
  def __call__(self, video):
    while video.shape[0] < self.output_size:
      zero_frame = torch.zeros(video.shape[1], video.shape[2], video.shape[3]) #padding frame
      #last_frame = video[video.shape[0]-1] #extract the last frame 
      video = torch.cat((video, zero_frame.unsqueeze(0)), 0) #add the padding frame at the end of the video
      #video = torch.cat((video, last_frame.unsqueeze(0)), 0) #add the padding frame at the end of the video

    return video



# Instantiation  of the dataloaders

"""
Function that instantiates the dataloaders of the training and test sets

:param batch_size: integer mini batch size used during training. Default = BATCH_SIZE.
:param video_root: string specifing the path to the dataset parent folder. Default = ds_path.
:return train_loader, test_loader: train loader, test loader.
"""
def get_data(batch_size = BATCH_SIZE, video_root = ds_path):
  
  # Prepare data transformations for the dataloaders
  transform = list()
  transform.append(T.Resize((256, 256)))                      
  transform.append(T.ToTensor())                              
  transform.append(T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]))  
  transform = T.Compose(transform)                
    
  # Load data
  train_dataset = VideoDataset(annotations_file = os.path.join(ds_path, "Train_Annotations.txt"), video_dir = video_root, video_transform = ZeroPadding(max_len()), frame_transform = transform)
  test_dataset = VideoDataset(annotations_file = os.path.join(ds_path, "Test_Annotations.txt"), video_dir = video_root, video_transform = ZeroPadding(max_len()), frame_transform = transform)

  # Initialize dataloaders
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle = False, drop_last = True)
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size, shuffle = False, drop_last = True) #None batch_size --> only one video is tested at a time
  
  return train_loader, test_loader


"""
--------------------------- Statistic ---------------------------
"""
"""
Function that output the scores distribution in both training and test sets.

:return test_dist, train_dist: scores distribution in test and train sets.
"""
def score_dist():
  test_dist = np.zeros(4) #vector that will contain the number of video for each score in the test set, i.e. test_dist[i] will contain the number of video with ground gruth label i
  train_dist = np.zeros(4) #vector that will contain the number of video for each score in the test set, i.e. train_dist[i] will contain the number of video with ground gruth label i

  with open(os.path.join(ds_path, "Test_Annotations.txt")) as f:
    for line in f.readlines():
      score = int(line.split(";")[1]) #extract the score
      
      # Fill test_dist
      test_dist[score] = test_dist[score] + 1
  f.close()

  with open(os.path.join(ds_path, "Train_Annotations.txt")) as f:
    for line in f.readlines():
      score = int(line.split(";")[1]) #extract the score

      # Fill test_dist
      train_dist[score] = train_dist[score] + 1

  f.close()

  return test_dist, train_dist



"""
--------------------------- Testing of the code above ---------------------------
"""
if __name__ == "__main__":
  from Data import *

  print("Patient used in the training set: ", p_train)
  print("Patient used in the test set: ", p_test)
  te_dist, tr_dist = score_dist()
  print("Scores distribution in the test set: ", te_dist)
  print("Scores distribution in the train set: ", tr_dist)

  ta, te = get_data(BATCH_SIZE, ds_path)
  print("\n\nTRAINING")
  for idx, (v, l, p) in enumerate(ta):
    if idx == 0:
      for video in v:
        print("Video shape: ", video.shape)

      print("Paths: ", p)
      print("Labels: ", l)
    else:
      break


  print("\n\nTEST")
  for idx, (v, l, p) in enumerate(te):
    if idx == 0:
      print("Video: ", video)
      print("Video shape: ", video.shape)

      print("Paths: ", p)
      print("Labels: ", l)
    else:
      break