import os
import file
from math import sqrt
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import low_res_analysis
from skimage.exposure import rescale_intensity

trans_mat_ini = [[-8.76320198e-03, 9.90991239e-01, 1.00157398e+05], [-1.00051713e+00,
 -3.53413164e-03,  3.77253688e+04], [  0.00000000e+00 , 0.00000000e+00,
  1.00000000e+00]]


x_min = [float("inf"), 0]
x_max = [float("inf") * -1, 0]
y_min = [0, float("inf")]
y_max = [0, float("inf") * -1]

actual_x_min = []
actual_x_max = []
actual_y_min = []
actual_y_max = []


pre_LCM_pts = []

valid = []

def write_point(point_id, f, x, y):
    x_tl = x - 50
    y_tl = y - 50
    x_br = x + 50
    y_br = y + 50
    
    f.write("\nRectangle	green	2	%d	1,1\n" % point_id)
    f.write(".	%.1f,%.1f\t%.1f,%.1f" %(x_tl, y_tl, x_br, y_br))


def get_points():
    
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    path = os.path.join(parent, "Manual", "abnormal.txt")
    
    f = open(path, "r")
    file_lines = f.readlines()
    
    for line in file_lines:
        valid.append(int(line))
    
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    path = os.path.join(parent, "Point_Lists", file.pre_LCM)
    
    f = open(path, "r")
    file_lines = f.readlines()
    
    for i in range(len(file_lines)):  
        
        if (i+1) in valid:
            entry = file_lines[i].split(" ")
        
            while '' in entry:
                entry.remove('')
            
            x_val = float(entry[0])
            y_val = float(entry[1])
            x_val_im = float(entry[2])
            y_val_im = float(entry[3])
            image_id = int(entry[4])
            
            v = np.dot(trans_mat_ini, [x_val, y_val, 1])
            
            x_val = v[0]
            y_val = v[1]
    
            pre_LCM_pts.append([x_val, y_val, x_val_im, y_val_im, image_id])        
    
def euc_distance(v1, v2):
    return sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)

def err(params, pred_coord, act_coord):
    a11, a12, a13, a21, a22, a23 = params

    trans_mat = np.array([[a11, a12, a13],
              [a21, a22, a23],
              [0,   0,   1  ]])
    
    x_min_error = euc_distance(act_coord[0], np.dot(trans_mat, np.asarray(pred_coord[0]).transpose()))
    x_max_error = euc_distance(act_coord[1], np.dot(trans_mat, np.asarray(pred_coord[1]).transpose()))
    y_min_error = euc_distance(act_coord[2], np.dot(trans_mat, np.asarray(pred_coord[2]).transpose()))
    y_max_error = euc_distance(act_coord[3], np.dot(trans_mat, np.asarray(pred_coord[3]).transpose()))

    return np.mean([x_min_error, x_max_error, y_min_error, y_max_error])

def generate_coord_LCM(cells):
    global x_min, x_max, y_min, y_max, actual_x_min, actual_x_max, actual_y_min, actual_y_max

    print("Generating Calibrated Coordinates...")
    pred_coord = [x_min[0:2] + [1], x_max[0:2] + [1], y_min[0:2] + [1], y_max[0:2] + [1]]
    act_coord = [actual_x_min, actual_x_max, actual_y_min, actual_y_max]
    
    trans_mat = minimize(err, x0 = [1, 1, 1, 1, 1, 1], args = (pred_coord, act_coord)).x.reshape(2,3)
    
    print(trans_mat)
        
    trans_mat = np.append(trans_mat, [0,0,1])
            
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    path = os.path.join(parent, 'Point_Lists', 'calibrated_points_for_LCM.txt')   
    

    f = open(path, "w")
    
    f.write("PALMRobo Elements\n")
    f.write("Version:	V 4.5.0.9\n")
    f.write("Date, Time :	24.04.2018	12:29:07\n\n")
    f.write("MICROMETER\n")
    f.write("Elements :\n\n")
    f.write("Type	Color	Thickness	No	CutShot	Area	Comment	Coordinates\n")
    
        
    for i in range(len(cells)):
        trans_v = np.dot(trans_mat.reshape(3,3), cells[i][0:2] + [1])
        write_point(i + 1, f, trans_v[0], trans_v[1])
    f.close()

def run_calibration_LCM(cells):
    global x_min, x_max, y_min, y_max, actual_x_min, actual_x_max, actual_y_min, actual_y_max
    
    for i in range(len(cells)):
        if (cells[i][0] < x_min[0]):
            x_min = cells[i]
        if (cells[i][1] < y_min[1]):
            y_min = cells[i]
        if (cells[i][0] > x_max[0]):
            x_max = cells[i]
        if (cells[i][1] > y_max[1]):
            y_max = cells[i]
            
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    path = os.path.join(parent, 'Point_Lists', 'calibration_points_for_LCM.pts')   

    f = open(path, "w")
    
#    write_point(path, 1, f, x_min[0], x_min[1])
#    write_point(path, 2, f, x_max[0], x_max[1])
#    write_point(path, 3, f, y_min[0], y_min[1])
#    write_point(path, 4, f, y_max[0], y_max[1])
    
    f.close()

                
    print("Go to: x = %.2f, y = %.2f" % (x_min[0], x_min[1]))
          
    camera = low_res_analysis.get_low_res_DAPI_image(file.DAPI_file % x_min[-1])
    
    fig, ax = plt.subplots(1, figsize = (10,10))
    c = plt.Circle((x_min[2], x_min[3]), 10, color = 'red', linewidth = 1, fill = False)
    ax.add_patch(c)

    ax.imshow(camera, cmap='gray', interpolation='nearest')
    ax.set_aspect('equal')
    plt.show()    
    
    x = input("Enter actual x coord: ")
    y = input("Enter actual y coord: ")
    
    actual_x_min = [float(x),float(y)]
    
    print("Go to: x = %.2f, y = %.2f" % (x_max[0], x_max[1]))
          
    camera = low_res_analysis.get_low_res_DAPI_image(file.DAPI_file % x_max[-1])
    
    
    fig, ax = plt.subplots(1, figsize = (10,10))
    c = plt.Circle((x_max[2], x_max[3]), 10, color = 'red', linewidth = 1, fill = False)
    ax.add_patch(c)
    

    ax.imshow(camera, cmap='gray', interpolation='nearest')
    ax.set_aspect('equal')
    plt.show()    
    
    x = input("Enter actual x coord: ")
    y = input("Enter actual y coord: ")
    
    actual_x_max = [float(x),float(y)]
    
    print("Go to: x = %.2f, y = %.2f" % (y_min[0], y_min[1]))
          
    camera = low_res_analysis.get_low_res_DAPI_image(file.DAPI_file % y_min[-1])
      
    fig, ax = plt.subplots(1, figsize = (10,10))
    c = plt.Circle((y_min[2], y_min[3]), 10, color = 'red', linewidth = 1, fill = False)
    ax.add_patch(c)

    ax.imshow(camera, cmap='gray', interpolation='nearest')
    ax.set_aspect('equal')
    plt.show()    
    
    x = input("Enter actual x coord: ")
    y = input("Enter actual y coord: ")
    
    actual_y_min = [float(x),float(y)]
    
    print("Go to: x = %.2f, y = %.2f" % (y_max[0], y_max[1]))
          
    camera = low_res_analysis.get_low_res_DAPI_image(file.DAPI_file % y_max[-1])
       
    fig, ax = plt.subplots(1, figsize = (10,10))
    c = plt.Circle((y_max[2], y_max[3]), 10, color = 'red', linewidth = 1, fill = False)
    ax.add_patch(c)

    ax.imshow(camera, cmap='gray', interpolation='nearest')
    ax.set_aspect('equal')
    plt.show()   
    
    x = input("Enter actual x coord: ")
    y = input("Enter actual y coord: ")
    
    actual_y_max = [float(x),float(y)]