import os
import file

pre_LCM_pts = []

def get_points():
    
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    path = os.path.join(parent, "Point_Lists", file.pre_LCM)
    
    f = open(path, "r")
    file_lines = f.readlines()
    
    for i in range(len(file_lines)):  
        entry = file_lines[i].split(" ")
        
        while '' in entry:
            entry.remove('')
        
        x_val = float(entry[1])
        y_val = float(entry[2])
        
        pre_LCM_pts.append([x_val, y_val])
    
    print(pre_LCM_pts)