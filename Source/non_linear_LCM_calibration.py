import os
import low_res_analysis
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
from scipy.optimize import minimize
import file
import itertools


x_min = [float("inf"), 0]
x_max = [float("inf") * -1, 0]
y_min = [0, float("inf")]
y_max = [0, float("inf") * -1]

actual_coordinates = [] # This list will eventually hold all of the user-guided corrected calibration point coordinates.

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
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10 = params # Need 10 params because 2nd degree in both x and y and have a constant.
    
    # Apply polynomial predictor function to each coordinate in pred_coord:
    pred_x_coord = c1*pred_coord[0]**2 + c2*pred_coord[0] + c3*pred_coord[1]**2 + c4*pred_coord[1] + c5
    pred_y_coord =c6*pred_coord[0]**2 + c7*pred_coord[0] + c8*pred_coord[1]**2 + c9*pred_coord[1] + c10
    calc_coord = list(zip(pred_x_coord, pred_y_coord)) # combine the predicted/calculated x and y values into a single list.

    pred_pts_error = [] # This will eventually hold the euc distance between the act_coord and the pred_coord
    for i in range(len(calc_coord)):
        pred_pts_error.append(euc_distance(act_coord[i], calc_coord[i]))

    mean_error = np.mean([pred_pts_error[i] for i in range(len(pred_pts_error))])
    return mean_error

def generate_coord(cells):
    # =============================================================================
    #     Input: list of lists (sublists have 5 elements: DV coordinates, image coordinates, and image ID)
    #     Output: calibrated pts written to file "LCM_calibrated_pts.pts"
    # =============================================================================
    print("Generating Calibrated Coordinates...")
    ## Think I need to add in global variables here. Note that Tiger doesn't include any global variables in the DV version of generate_coords, and it still works.
    global actual_coordinates
    
    pred_coord = []
    for i in range(len(actual_coordinates)):
        pred_coord.append(actual_coordinates[i][0:2]) # Grab just the x and y coordinate from actual_coordinates
    
    act_coord = [actual_pt_1, actual_pt_2, actual_pt_3, actual_pt_4, actual_pt_5, actual_pt_6, actual_pt_7, actual_pt_8, actual_pt_9]
    
    best_parameters = minimize(err, x0 = [1 for i in range(9)], args = (pred_coord, act_coord)).x.reshape(2,5) # Not sure if it's a good idea to initialize with all 1s. Can we make a better guess?
    
    # Apply the optimized polynomial parameters to each element in cells, and write the points to the calibrated pts file.    
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    path = os.path.join(parent, "Point_Lists", "LCM_calibrated_pts.pts")
        
    f = open(path, "w")
    
    f.write("PALMRobo Elements\n")
    f.write("Version:	V 4.5.0.9\n")
    f.write("Date, Time :	24.04.2018	12:29:07\n\n")
    f.write("MICROMETER\n")
    f.write("Elements :\n\n")
    f.write("Type	Color	Thickness	No	CutShot	Area	Comment	Coordinates\n")

    for i in range(len(cells)):
        x = cells[i][0] # access the x-coordinate
        y = cells[i][1] # access the y-coordinate
        cell_poly = [x**2, x, y**2, y, 1] # this is the backbone of the polynomial we are using currently.
        poly_pt = np.dot(best_parameters, cell_poly) # multiplication gives polynomial prediction for x and y coords. Could also use matmul. 
        write_point(path, i+1, f, poly_pt[0], poly_pt[1])
        
    f.close()
   
    
def display_plot_image(camera_image, coordinate):
    # =============================================================================
    #     Inputs: takes the image and a single image/object coordinate.
    #     Output: image with circle plotted around the object in the image
    # =============================================================================            
    fig, ax = plt.subplots(1, figsize = (15,15))
    c = plt.Circle((coordinate[2], coordinate[3]), 10, color = 'red', linewidth = 1, fill = False)
    ax.add_patch(c)

    ax.imshow(camera_image, cmap='gray', interpolation='nearest')
    ax.set_aspect('equal')
    plt.axis('off')
    plt.show()  
    
def run_calibration(cells):
    # =============================================================================
    # Input:
    #     cells is a list of lists where each sublist has 5 elements: coordinates from DV, coordinates for image, then ID number
    # 
    # Output:
    #     actual coordinates based on user input while re-centering on the LCM. 
    #     This is returned as a list called actual_coordinates. 
    # =============================================================================
    global x_min, x_max, y_min, y_max, actual_coordinates
    
    # Find the cells needed to create a grid encompassing all of the cells on the slide.
    for i in range(len(cells)):
        if (cells[i][0] < x_min[0]):
            x_min = cells[i] 
        if (cells[i][1] < y_min[1]):
            y_min = cells[i]
        if (cells[i][0] > x_max[0]):
            x_max = cells[i]
        if (cells[i][1] > y_max[1]):
            y_max = cells[i]

    # Now have the points needed to make a large rectangule encompassing all of the cells. We use this rectangle to methodically select 9 points in a grid.
    # These two variables define the middle axes. We will look for cells nearest these x and y values so we get as evenly-spread sampling as possible across the slide.   
    # Currently set up so 5 vertical divisions and 3 horizontal divisions => total of 15 grid points used to calibrate.
    near_x_min = x_min[0]
    near_x_half = [np.average([x_min[0], x_max[0]])]
    near_x_qtr = [np.average([x_min[0], near_x_half[0]])]
    near_x_three_qtr = [np.average([x_max[0], near_x_half[0]])]
    near_x_max = x_max[0]
    near_y_min = y_min[1]
    near_y_mid = [np.average([y_min[1], y_max[1]])]
    near_y_max = y_max[1]
    
    # Add all of the points that will be used to define the calibration grid to separate lists, one for the x positions and a second for the y positions.
    # The total number of calibration points is len(x_positions)*len(y_positions)
    x_positions = [x_min, near_x_qtr, near_x_half, near_x_three_qtr, x_max]
    y_positions = [y_min, near_y_mid, y_max]
    
    # Make all of the grid coordinates from a combination of the x and y positions.
    # I do acknowledge that this method re-locates the 4 corner cells, and thus is a little redundant. However, using itertools makes it easier to modify the grid size later.
    remaining_grid_pts = list(itertools.product(x_positions, y_positions)) # Note that this returns list of tuples, so these points are immutable.
    
    # Search through list of cell coordinates, "cells," to find cells near the grid points.
    # That is, we want to find cells with the minimum euc distance to the remaining grid points.  
    calibration_points = []
    min_dist_cell_coords = []
    for i in range(len(remaining_grid_pts)):
        # Find the cell with the minimum euc distance to the grid coordinate at hand.
        nearest = min(cells, key=lambda x: euc_distance(x, remaining_grid_pts[i]))
        min_dist_cell_coords.append(nearest)     # In theory, "nearest" should have 5 elements. Need to check this.
        calibration_points.append(min_dist_cell_coords[i][0:2]) # Add the cell coordinates for each calibration point to a list.
        nearest = 0
    
     
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    path = os.path.join(parent, "Point_Lists", "calibration_pts_for_LCM.pts")
    
    f = open(path, "w")
    
    # Write the calibration points to the calibration points file.
    for i in range(len(calibration_points)):
        write_point(path, i+1, f, calibration_points[i][0], calibration_points[i][1])
        

    f.close()
    
    # The below provides user input from the LCM to calibrate the coordinates for each of the tested points.
    actual_coordinates = []
    for i in range(len(calibration_points)):
        print("Go to: x = %.2f, y = %.2f" % (calibration_points[i][0], calibration_points[i][1]) # Go to the first calibration point, which is the upper-leftmost cell.          

        camera = low_res_analysis.get_low_res_DAPI_image(file.DAPI_file % calibration_points[i][-1])
        display_plot_image(camera, calibration_points[i])
        
        x = input("Enter actual x coord: ")
        y = input("Enter actual y coord: ")
        
        actual_coordinates.append([float(x),float(y)])



    

    

    
    
        
        
