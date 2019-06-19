import os 
import shutil
from dataset_functions import align_dataset_mtcnn, tracking

if os.path.isdir('dataset'):
  shutil.rmtree('dataset')
os.mkdir('dataset')

for trainfolder in os.listdir('raw_dataset'):
  if 'DS_Store' in trainfolder:
    continue
  for day in os.listdir('raw_dataset/' + trainfolder):  
    if 'DS_Store' in day:
      continue
    print(trainfolder + '/' + day)
    align_dataset_mtcnn(trainfolder + '/' + day, 'dataset')
    # if result != None:  
    #   print(result)
    #   tracking(result[0], 'dataset', result[1], trainfolder + '/' + day)