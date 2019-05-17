# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
def get_dataset(path, has_class_directories=True):
	# expanduser is not worked in mac
	dataset = []
	# path_exp = os.path.expanduser(path)
	classes = [p for p in os.listdir(path) \
					if os.path.isdir(os.path.join(path, p))]
	classes.sort()
	# if 'bounding_boxes_txt' in data_dir_names:
	# 		data_dir_names.remove('bounding_boxes_txt')
	nrof_classes = len(classes)
	for i in range(nrof_classes):
		class_name = classes[i]
		facedir = os.path.join(path, class_name)
		image_paths = get_image_paths(facedir)
		dataset.append(ImageClass(class_name, image_paths))
  
	return dataset

def get_image_paths(facedir):
	image_paths = []
	if os.path.isdir(facedir):
		images = os.listdir(facedir)
		image_paths = [os.path.join(facedir,img) for img in images]
	return image_paths

class ImageClass():
	"Stores the paths to images for a given class"
	def __init__(self, name, image_paths):
		self.name = name
		self.image_paths = image_paths
  
	def __str__(self):
		return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
	def __len__(self):
		return len(self.image_paths)