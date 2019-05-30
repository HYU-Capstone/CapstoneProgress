import os 
import shutil
from functions import align_dataset_mtcnn, tracking

if os.path.isdir('testset'):
  shutil.rmtree('testset')
os.mkdir('testset')

for testfolder in os.listdir('raw_testset'):
  if 'DS_Store' in testfolder:
    continue
  print(testfolder)
  result = align_dataset_mtcnn(testfolder, 'testset')
  if result != None:  
    print(result)
  tracking(result[0], 'testset', result[1], testfolder)