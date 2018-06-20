import os

DAPI_file = "DAPI_atubulin_20x_1_R3D_PRJ_w435_t%.3d.tif"
pattern_file = "DAPI_atubulin_20x_1_R3D_PRJ_w676_t%.3d.tif"
low_res_log = "DAPI_atubulin_20x_1_R3D.dv.log"

hr_DAPI = "20180607_low_circle_1_DAPI_a_tubulin_CEP135_pattern_03_%.2d_R3D_D3D_PRJ_w435.tif"
hr_atubulin = "20180607_low_circle_1_DAPI_a_tubulin_CEP135_pattern_03_%.2d_R3D_D3D_PRJ_w523.tif"
hr_pattern = "DAPI_atubulin_pattern_all_1_%.3d_R3D_D3D_PRJ_w676.tif"


pre_LCM = "pre_LCM"

num_low_res = 608
num_high_res = 127

def get_low_res_coord():
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    # path = os.path.join(parent, "Low_Res_Input_Images", low_res_log) # for 10x imaging
    path = os.path.join(parent, "Low_Res_Input_Images_20x", low_res_log) # for 20x imaging
    
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