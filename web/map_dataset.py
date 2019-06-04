from functions import connect_db, create_user
import os
import shutil 

for name in os.listdir('TRAIN'):
    if name == '.DS_Store':
        continue
    uid = create_user(name, name +'@gmail.com')
    shutil.move('TRAIN/' + name, 'TRAIN/' + str(uid))