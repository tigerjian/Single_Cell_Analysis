import os
import file
from math import sqrt
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import low_res_analysis
from skimage.exposure import rescale_intensity
import itertools

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

grid_size = [5,3] # larger side first

pre_LCM_pts = []

valid = []

cali_cells = []
act_cells = []

def display_plot_image(image_file, cell):
    image_s = low_res_analysis.image_size
    camera = low_res_analysis.get_low_res_DAPI_image(image_file % cell[-1])
    # to print out according to field of view on LCM
    camera = np.flipud(camera)
    camera = np.rot90(camera)

    fig, ax = plt.subplots(1, figsize = (10,10))
    c = plt.Circle((image_s - cell[3], image_s - cell[2]), 10, color = 'red', linewidth = 1, fill = False)
    ax.add_patch(c)
    ax.imshow(camera, cmap='gray', interpolation='nearest')
    ax.set_aspect('equal')
    plt.show()   
    
def write_rectangle(point_id, f, x, y):
    x_tl = x - 50
    y_tl = y - 50
    x_br = x + 50
    y_br = y + 50
    
    f.write("\nRectangle	green	2	%d	1,1\n" % point_id)
    f.write(".	%.1f,%.1f\t%.1f,%.1f" %(x_tl, y_tl, x_br, y_br))
    
def write_point(point_id, f, x, y):
    f.write("\nDot\tyellow	\t3\t%d\t0,0\t0.000" % point_id)
    f.write("\n.	%.1f,%.1f\n" % (x,y))

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
    return sqrt((float(v1[0]) - float(v2[0]))**2 + (float(v1[1]) - float(v2[1]))**2)

def find_closest_pt(pt, pt_list):
    dist = float("inf")
    
    for p in pt_list:
        d = euc_distance(pt, p)
        if d < dist:
            dist = d
            closest_pt = p
    return closest_pt


def err(params, cali_cells, act_cells):
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = params # Need 10 params because 2nd degree in both x and y and have a constant.
    
    trans_mat = np.array([[c1, c2, c3, c4, c5], [c6, c7, c8, c9, c10]])
    
    dist = []
    
    for i in range(len(cali_cells)):
        x = cali_cells[i][0]
        x_2 = x**2
        y = cali_cells[i][1]
        y_2 = y**2
        
        cal_v = [x_2, y_2, x, y, 1]
        pred_cell_coord = np.dot(trans_mat, cal_v)

        dist.append(euc_distance(pred_cell_coord, act_cells[i]))
        
    return np.mean(dist)

def generate_coord_LCM():
    global cali_cells, act_cells
    
    m = minimize(err, x0 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1], args = (cali_cells, act_cells), method = 'Powell')
    trans_mat = m.x.reshape(2,5)
    print("### Trans Mat ###")
    print(m)
    print("#################")
    
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    path = os.path.join(parent, 'Point_Lists', 'calibrated_points_for_LCM.txt')   
    
    f = open(path, "w")
    
    f.write("PALMRobo Elements\n")
    f.write("Version:	V 4.5.0.9\n")
    f.write("Date, Time :	24.04.2018	12:29:07\n\n")
    f.write("MICROMETER\n")
    f.write("Elements :\n\n")
    f.write("Type	Color	Thickness	No	CutShot	Area	Comment	Coordinates\n")
    
    for i in range(len(cali_cells)):
        x = cali_cells[i][0]
        x_2 = x**2
        y = cali_cells[i][1]
        y_2 = y**2
        
        cal_v = [x_2, y_2, x, y, 1]
        pred_cell_coord = np.dot(trans_mat, cal_v)
        
        print(pred_cell_coord[0], "\t", pred_cell_coord[1])
    f.close()

def run_calibration_LCM(cells):
# =============================================================================
#     Input: 
#         coordinates of cells to be calibrated in the following format,
#     
#     DV x coord, DV y coord, image x coord, image y coord, image id
#     
#     Output:
#         performs a grid calibration and obtains the appropriate correct coordinate
#         for each point
# =============================================================================
    global x_min, x_max, y_min, y_max, actual_x_min, actual_x_max, actual_y_min, actual_y_max    
    global cali_cells, act_cells
    
    for i in range(len(cells)):
        # Note: y is the longer side of slide
        if (cells[i][0] < x_min[0]):
            x_min = cells[i]
        if (cells[i][1] < y_min[1]):
            y_min = cells[i]
        if (cells[i][0] > x_max[0]):
            x_max = cells[i]
        if (cells[i][1] > y_max[1]):
            y_max = cells[i]
            
    x_grid_points = np.linspace(x_min[0], x_max[0], grid_size[1])
    y_grid_points = np.linspace(y_min[1], y_max[1], grid_size[0])

    grid_points = list(itertools.product(x_grid_points, y_grid_points))
    
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    path = os.path.join(parent, 'Point_Lists', 'calibration_points_for_LCM.txt')   
    
    for i in range(len(grid_points)):
        closest_pt = find_closest_pt(grid_points[i], pre_LCM_pts)
        
        if closest_pt not in cali_cells:
            cali_cells.append(closest_pt)
        
    f = open(path, "w")
    
    f.write("PALMRobo Elements\n")
    f.write("Version:	V 4.5.0.9\n")
    f.write("Date, Time :	24.04.2018	12:29:07\n\n")
    f.write("MICROMETER\n")
    f.write("Elements :\n\n")
    f.write("Type	Color	Thickness	No	CutShot	Area	Comment	Coordinates\n")
    
    for i in range(len(cali_cells)):
        write_point(i + 1, f, cali_cells[i][0], cali_cells[i][1])
        
    f.close()
    
    for i in range(len(cali_cells)):
        print("Reference image for x = %.2f, y = %.2f" % (cali_cells[i][0], cali_cells[i][1]))
        display_plot_image(file.DAPI_file, cali_cells[i])
        
    for i in range(len(cali_cells)):
        print("Enter actual coordinates for x = %.2f, y = %.2f" % (cali_cells[i][0], cali_cells[i][1]))
        actual_x = input("x: ")
        actual_y = input("y: ")
        
        act_cells.append([actual_x, actual_y])
    
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
