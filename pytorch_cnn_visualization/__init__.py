from .visualize import Visualize

import os, json

class_name_file = open(os.path.join(os.path.dirname(__file__), 'data/class_name.json'), 'r')
class_name = json.load(class_name_file)
class_name_file.close()
