import os 
def get_base_dir():
    this_file = os.path.abspath(__file__)
    base_dir = os.path.dirname(this_file)   # .../nanoforge/src
    return base_dir