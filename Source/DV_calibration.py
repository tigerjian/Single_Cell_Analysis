import os
import low_res_analysis
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
from scipy.optimize import minimize
import file


x_min = [float("inf"), 0]
x_max = [float("inf") * -1, 0]
y_min = [0, float("inf")]
y_max = [0, float("inf") * -1]

actual_x_min = []
actual_x_max = []
actual_y_min = []
actual_y_max = []

def write_point(path, point_id, f, x, y):
    p_str = ""
    p_str += " " * (4 - len(str(point_id)))
    p_str += "%d:" % point_id
    
    if (x >= 0):
        p_str += " " * (10 - len("%.2f" % x))
        p_str += "+"
        p_str += "%.2f" % x
    else:
        p_str += " " * (11 - len("%.2f" % x))
        p_str += "%.2f" % x
        
    if (y >= 0):
        p_str += " " * (10 - len("%.2f" % y))
        p_str += "+"
        p_str += "%.2f" % y
    else:
        p_str += " " * (11 - len("%.2f" % y))
        p_str += "%.2f" % y
        
    p_str += " " * 3
    p_str += "+0.00"
        
    f.write(p_str + "\n")
    
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

def generate_coord(cells):
    print("Generating Calibrated Coordinates...")
    pred_coord = [x_min[0:2] + [1], x_max[0:2] + [1], y_min[0:2] + [1], y_max[0:2] + [1]]
    act_coord = [actual_x_min, actual_x_max, actual_y_min, actual_y_max]
    
    trans_mat = minimize(err, x0 = [1, 1, 1, 1, 1, 1], args = (pred_coord, act_coord)).x.reshape(2,3)
    trans_mat = np.append(trans_mat, [0,0,1])
    
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    path = os.path.join(parent, "Point_Lists", "DV_calibrated_pts.pts")
        
    f = open(path, "w")
        
    for i in range(len(cells)):
        trans_v = np.dot(trans_mat.reshape(3,3), cells[i][0:2] + [1])
        write_point(path, i + 1, f, trans_v[0], trans_v[1])
    f.close()
   
def run_calibration(cells):
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
    path = os.path.join(parent, "Point_Lists", "calibration_pts_for_DV.pts")
    
    f = open(path, "w")
    
    write_point(path, 1, f, x_min[0], x_min[1])
    write_point(path, 2, f, x_max[0], x_max[1])
    write_point(path, 3, f, y_min[0], y_min[1])
    write_point(path, 4, f, y_max[0], y_max[1])
    
    f.close()
    
                
    print("Go to: x = %.2f, y = %.2f" % (x_min[0], x_min[1]))
          
    camera = low_res_analysis.get_low_res_DAPI_image(file.DAPI_file % x_min[-1])
            
    fig, ax = plt.subplots(1, figsize = (15,15))
    c = plt.Circle((x_min[2], x_min[3]), 10, color = 'red', linewidth = 1, fill = False)
    ax.add_patch(c)

    ax.imshow(camera, cmap='gray', interpolation='nearest')
    ax.set_aspect('equal')
    plt.axis('off')
    plt.show()    
    
    x = input("Enter actual x coord: ")
    y = input("Enter actual y coord: ")
    
    actual_x_min = [float(x),float(y)]
    
    print("Go to: x = %.2f, y = %.2f" % (x_max[0], x_max[1]))
          
    camera = low_res_analysis.get_low_res_DAPI_image(file.DAPI_file % x_max[-1])
            
    fig, ax = plt.subplots(1, figsize = (15,15))
    c = plt.Circle((x_max[2], x_max[3]), 10, color = 'red', linewidth = 1, fill = False)
    ax.add_patch(c)

    ax.imshow(camera, cmap='gray', interpolation='nearest')
    ax.set_aspect('equal')
    plt.axis('off')
    plt.show()    
    
    x = input("Enter actual x coord: ")
    y = input("Enter actual y coord: ")
    
    actual_x_max = [float(x),float(y)]
    
    print("Go to: x = %.2f, y = %.2f" % (y_min[0], y_min[1]))
          
    camera = low_res_analysis.get_low_res_DAPI_image(file.DAPI_file % y_min[-1])
            
    fig, ax = plt.subplots(1, figsize = (15,15))
    c = plt.Circle((y_min[2], y_min[3]), 10, color = 'red', linewidth = 1, fill = False)
    ax.add_patch(c)

    ax.imshow(camera, cmap='gray', interpolation='nearest')
    ax.set_aspect('equal')
    plt.axis('off')
    plt.show()    
    
    x = input("Enter actual x coord: ")
    y = input("Enter actual y coord: ")
    
    actual_y_min = [float(x),float(y)]
    
    print("Go to: x = %.2f, y = %.2f" % (y_max[0], y_max[1]))
          
    camera = low_res_analysis.get_low_res_DAPI_image(file.DAPI_file % y_max[-1])
            
    fig, ax = plt.subplots(1, figsize = (15,15))
    c = plt.Circle((y_max[2], y_max[3]), 10, color = 'red', linewidth = 1, fill = False)
    ax.add_patch(c)

    ax.imshow(camera, cmap='gray', interpolation='nearest')
    ax.set_aspect('equal')
    plt.axis('off')
    plt.show()   
    
    x = input("Enter actual x coord: ")
    y = input("Enter actual y coord: ")
    
    actual_y_max = [float(x),float(y)]
    
    
        
        
