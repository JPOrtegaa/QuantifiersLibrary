import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))

root_dir = os.path.abspath(os.path.join(cur_dir, '..\..'))

sys.path.append(root_dir)