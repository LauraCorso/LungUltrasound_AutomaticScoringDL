# MEDICAL IMAGING DIAGNOSTIC PROJECT - LengthAnalysis.py
# CORSO LAURA 230485

"""
Code to analyze what is the best approach to get videos all of the same length in a batch.

In order to create the bathces at the training step, the videos have to be all of the same length.
Since this is not the case for the given dataset, two analysis are done:
* lengths vs scores: as can be seen from the graphs produced the scores are well distributed along the lengths. 
                     In fact, both the smallest  (len < 91) and the greater (len > 91) lengths are quite uniformly distributed in order to cover all the four scoures.

* lengths vs patients: as can be seen from the produced graphs, the patients n.1051 and n.1047 are the ones with the shortest videos.

In order to avoid biases in the training of the network and simultaneously solve the problem of different lenghts, a zero-padding techniques is used on the videos of the training set. 
In addition, this set won't contain the patients n.1051 and n.1047 which will be located in the test set.
"""


"""
--------------------------- Imports ---------------------------
"""
from Config import *
from Data import *


"""
--------------------------- Creation of a list of tuples (patient_id, video_length, score) ---------------------------
"""
#prepare data transformations for the dataloader
transform = list()
transform.append(T.Resize((256, 256)))                      
transform.append(T.ToTensor())                              
transform.append(T.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]))  
transform = T.Compose(transform)  

#instatiation of the dataloader
dataset = VideoDataset(annotations_file = os.path.join(ds_path, "Annotations.txt"), video_dir = ds_path, video_transform = None, frame_transform = transform)

l = [] #list of tuples (patient_id, video_length, score)
patient_ids = set() #set of patient_ids
lengths = set() #set of video lengths

for video, label, path in dataset:
  patient_id = path.split("/")[-3]
  patient_ids.add(patient_id)
  lengths.add(video.shape[0])
  l.append((patient_id, video.shape[0], label))


"""
--------------------------- Length vs score ---------------------------
"""
# Creation of a folder in root_path to contain the produced graphs
if not os.path.exists(result_path):
    os.mkdir(result_path)

if not os.path.exists(os.path.join(result_path, "LenghtVScore")):
    os.mkdir(os.path.join(result_path, "LenghtVScore"))

# Computation of the graphs
for idx, length in enumerate(sorted(lengths)):
  filt_l = filter(lambda v: v[1] == length, l) #filters the list extracting the videos with the same length
  new_l = list(filt_l)

  for score in range(4):
    filt_s = filter(lambda x: x[2] == score, new_l) #filter the list extracting the videos for each score
    
    #assign the number of videos for each score
    if score == 0:
      score_0 = len(list(filt_s))

    elif score == 1:
      score_1 = len((list(filt_s)))

    elif score == 2:
      score_2 = len((list(filt_s)))

    else:
      score_3 = len((list(filt_s)))

  #visualization of the bar plot (one plot for each lenght)
  fig = plt.figure(idx)
  plt.bar(list(np.arange(0, 4, 1)), list([score_0, score_1, score_2, score_3]))
  plt.xlabel("Scores")
  plt.ylabel("N. of video")
  t = "Length " + str(length)
  plt.title(t)
  plt.xticks(range(0, 4))
  m = max(list([score_0, score_1, score_2, score_3]))+1
  plt.yticks(range(0, m))
  plt.show(block = False)
  t = t + ".png"
  fig.savefig(os.path.join(root_path, "LenghtVScore", t))


"""
--------------------------- Patient vs length ---------------------------
"""
# Creation of a folder in root_path to contain the produced graphs
if not os.path.exists(result_path):
    os.mkdir(result_path)

if not os.path.exists(os.path.join(result_path, "LenghtVPatient")):
    os.mkdir(os.path.join(result_path, "LenghtVPatient"))

n_l = len(lengths)

# Computation of the graphs
for idx, patient in enumerate(patient_ids):
  filt_p = filter(lambda v: v[0] == patient, l) #filters the list extracting the videos for every patient
  new_l = list(filt_p)

  n_videos_l = [] #list that contains the number of videos for every length found for the patient 
  len_p = [] #list that contains the video lengths for the given patient
  for length in sorted(lengths):
    filt_l = filter(lambda x: x[1] == length, new_l) #filter the list extracting the videos for each length
    final_l = list(filt_l)

    #select only the lengths found for the patient
    if len(final_l) > 0:
      n_videos_l.append(len(final_l))
      len_p.append(length)

  #visualization of the bar plot (one plot for each patient)
  fig = plt.figure(idx+n_l, figsize = (25, 5))
  plt.bar(len_p, n_videos_l)
  plt.xlabel("Lengths")
  plt.ylabel("N. of video")
  t = "Patient " + str(patient)
  plt.title(t)
  plt.xticks(range(min(len_p), max(len_p)+1))
  m = max(n_videos_l)+1
  plt.yticks(range(0, m))
  plt.xticks(
      rotation = 90, 
      fontweight = 'light',
      fontsize = 'x-large'  
  )
  plt.show(block = False)
  t = t + ".png"
  fig.savefig(os.path.join(root_path, "LenghtVPatient", t))