import os

DAPI_file = "DAPI_atubulin_pattern_1_R3D_PRJ_w435_t%.3d.tif"



def get_low_res_coord():
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    path = os.path.join(parent, "Low_Res_Input_Images", "DAPI_atubulin_pattern_1_R3D.dv.log")
    
    file = open(path, "r")
    file_lines = file.readlines()
         
    coord = {}
    
    entry_id = 1

    for i in range(10, len(file_lines)):
        if ("Stage coordinates:") in file_lines[i]:
            x_coord = file_lines[i].split(',')[0][33:]
            y_coord = file_lines[i].split(',')[1]
            entry = [x_coord, y_coord]
            if entry not in coord.values():
                coord[entry_id] = entry
                entry_id += 1
    return coord