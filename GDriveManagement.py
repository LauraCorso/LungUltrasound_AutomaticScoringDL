# MEDICAL IMAGING DIAGNOSTIC PROJECT - GDriveManagement.py
# CORSO LAURA 230485

"""
Code to manage the dataset on GDrive (to execute into a Colab file).

This file contains the code that is used to prepare the data of the dataset Dati San Matteo 2 in a compressed format in order to use them on the Azure machine. 
In this file it is also done the detection of mismatches between the files of the dataset and the SanMatteo.mat. The found mismatches are here corrected inserting the missing files.

Notes:  
*   the annotation file is done in MATLAB.
*   in order to find the mismatches, the fromMatToTxtIF.mat has to be executed in order to obtain mIF.txt.
"""


"""
--------------------------- Imports ---------------------------
"""
from Config import *


"""
--------------------------- Mounting of the Google Drive folder and creation of the needed folders ---------------------------
"""
from google.colab import drive
drive.mount('/content/gdrive')

# Folder for the dataset
if not os.path.exists(os.path.join(root_path_GDrive, "Dataset/")):
    os.mkdir(os.path.join(root_path_GDrive, "Dataset/"))



"""
--------------------------- Inconsistency finder ---------------------------

This section detect the mismatches between the files already present in the given dataset and the ones listed in SanMatteo.mat.
"""
# Inconsistency finder

#creation of a Dataframe containing in each row the patient id, the exam id and the area id
files = pd.DataFrame(columns = ["patient_id", "exam_id", "area_id"])
for name in os.listdir(Pdataset_GDrive):
  if name.startswith("1"):
    for exam in os.listdir(os.path.join(Pdataset_GDrive, name)):
      for file in os.listdir(os.path.join(Pdataset_GDrive, name, exam)):
        if file.endswith(".avi"):
          file = file[:-4]
          files.loc[len(files.index)] = [name, exam, file.split("_")[3]]

#extraction of patient id, exam id and area id from SanMatteo.mat
mat = pd.read_csv(os.path.join(root_path_GDrive, "mIF.txt"), sep = ";", names = ["patient_id", "exam_id", "area_id"])

#parsing
f = set(tuple(row) for row in files.values.astype(int).tolist())
m = set(tuple(row) for row in mat.values.tolist())
mismatches = sorted(m - f)



# Extraction and storage of the images from the .mat files 

for name in os.listdir(Pdataset_GDrive):
  #select only the folders containing the .mat files
  if name.startswith("1"):
    #create the folder for the patient
    if not os.path.exists(os.path.join(root_path_GDrive, "Dataset/", name)):
      os.mkdir(os.path.join(root_path_GDrive, "Dataset/", name))
    for exam in os.listdir(os.path.join(Pdataset_GDrive, name)):
      #create the folder for the exam
      if not os.path.exists(os.path.join(root_path_GDrive, "Dataset/", name, exam)):
        os.mkdir(os.path.join(root_path_GDrive, "Dataset/", name, exam))

      #check whether the exam has area mismatch (one exam can have more than one area mismatch)
      m = tuple(filter(lambda item: item[0:2] == (int(name), int(exam)), mismatches))
      if len(m) > 0:
        for i in range(len(m)):
          #create the folder for the area
          if not os.path.exists(os.path.join(root_path_GDrive, "Dataset/", name, exam, str(m[i][2]))):
            os.mkdir(os.path.join(root_path_GDrive, "Dataset/", name, exam, str(m[i][2])))

          #compose the name of the missing video
          filen = "clipped_" + exam + "_" + str(m[i][2]) + ".mp4"

          #read the missing video
          video = cv2.VideoCapture(os.path.join(root_path_GDrive, "Mismatches/", filen))

          #read the frames of the missing video
          fc = 0
          while(True):
            ret, frame = video.read()

            #if video is still left continue creating images
            if ret:
              f = "Frame_" + str(fc)
              save_path = os.path.join(root_path_GDrive, "Dataset/", name, exam, str(m[i][2]), f) + ".jpg"
              
              #writing the extracted images
              cv2.imwrite(save_path, frame)
              fc += 1

            else:
              break
          
          video.release()

      for file in os.listdir(os.path.join(Pdataset_GDrive, name, exam)):
        #select the .mat file containg the frames
        if re.match(".*[0-9]+.mat", file):
          mat_contents = sio.loadmat(os.path.join(Pdataset_GDrive, name, exam, file))

          #create the folder for the area
          if not os.path.exists(os.path.join(root_path_GDrive, "Dataset/", name, exam, file.split("_")[3][:-4])):
            os.mkdir(os.path.join(root_path_GDrive, "Dataset/", name, exam, file.split("_")[3][:-4]))

          #save all the frames as .jpg files
          for i in tqdm(range(mat_contents['frames'].shape[3]), desc = 'frames'):
            f = "Frame_" + str(i)
            save_path = os.path.join(root_path_GDrive, "Dataset/", name, exam, file.split("_")[3][:-4], f) + ".jpg"
            plt.imsave(save_path, mat_contents['frames'][:, :, :, i])
  
    #creation of a .zip archive for each patient
    shutil.make_archive(os.path.join(root_path_GDrive, "DatasetC/", name), 'zip', os.path.join(root_path_GDrive, "Dataset/", name))
    download(os.path.join(root_path_GDrive, "DatasetC/", name)+ ".zip")