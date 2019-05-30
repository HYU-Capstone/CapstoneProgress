import os 
import shutil
from functions import align_dataset_mtcnn, tracking

if os.path.isdir('dataset'):
  shutil.rmtree('dataset')
os.mkdir('dataset')

for trainfolder in os.listdir('raw_dataset'):
  if 'DS_Store' in trainfolder:
    continue
  print(trainfolder)
  result = align_dataset_mtcnn(trainfolder, 'dataset')
  if result != None:  
    print(result)
    tracking(result[0], 'dataset', result[1], trainfolder)