import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
import os 
from skimage import io
import file

import pdb

## Parameters for interactive plot.
# Not sure how to initialize the global variables for line and fig:
line = []
fig = []
ID = [] # Where will this be passed in from?
img_zoom = .7
arrow_length = img_zoom*150



def display_image(image):
    fig, ax = plt.subplots(1, figsize = (5,5))
    ax.imshow(image, cmap='gray', interpolation='nearest')
    ax.set_aspect('equal')
    plt.axis('off')
    plt.show()      

def display_color_coded_plot(x, y, cluster_labels, color_map_name, fig_size):
    ## Display color-coded t-SNE plot
    plt.figure(figsize = fig_size)
    plt.axis('equal')
    # scatter with colormap mapping to the cluster labels like those from DEC or k-means (returned as array)
    color_map = plt.get_cmap(color_map_name)
    
    max_label = max(cluster_labels)
    min_label = min(cluster_labels)
    if min_label != max_label:
        num_clusters = max_label - min_label + 1 # assumes cluster labels are in 1-unit increments 
    else:
        num_clusters = 1
        
    if max_label != min_label:
        norm = matplotlib.colors.BoundaryNorm(np.arange(min_label,num_clusters+1,1), color_map.N) # color bar from min_label up to the number of clusters
        plt.scatter(x,y,s=25,c=cluster_labels, marker = 'o', cmap = color_map, norm = norm );
        plt.colorbar()
    else:
        plt.scatter(x,y,s=25,c=cluster_labels, marker = 'o', cmap = color_map );
#    plt.show()      
    return plt


def get_image(name):
# =============================================================================
#     Gets a processed high res image from folder
#     and returns it as a matrix
# =============================================================================    
    parent = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    image_path = os.path.join(parent, "High_Res_Input_Images_Processed", name)
    image = io.imread(image_path)        
    return image



########### Starting code for interactive plot
# create the annotations box for the interactive plot
#pdb.set_trace()
DAPI_pic = get_image("DAPI_1.tif") ### hard-coded
im = OffsetImage(DAPI_pic, zoom=img_zoom, cmap = 'gray') # think all I need to do is pass in one DAPI picture
xybox=(arrow_length, arrow_length)
ab = AnnotationBbox(im, (0,0), xybox=xybox, xycoords='data',
            boxcoords="offset points",  pad=0.3,  arrowprops=dict(arrowstyle="->"))
    
    
def hover(event):
    global line, fig, images, ID
    global x, y
    
    # if the mouse is over the scatter points
    if line.contains(event)[0]:
        # find out the index within the array from the event
        ind, = line.contains(event)[1]["ind"]
        # get the figure size
        w,h = fig.get_size_inches()*fig.dpi
        ws = (event.x > w/2.)*-1 + (event.x <= w/2.) 
        hs = (event.y > h/2.)*-1 + (event.y <= h/2.)
        # if event occurs in the top or right quadrant of the figure,
        # change the annotation box position relative to mouse.
        ab.xybox = (xybox[0]*ws, xybox[1]*hs)
        # make annotation box visible
        ab.set_visible(True)
        # place it at the position of the hovered scatter point
        ab.xy =(x[ind], y[ind])
        
        # set the image corresponding to that point        
        print("ID[ind]:",ID[ind])
        name = "DAPI_%d.tif" % ID[ind]
        image_at_point = get_image(name)  
#        display_image(image_at_point) # The image looks fine, and is in greyscale.
        im.set_data(image_at_point) # Thus, this line must be the problem.
        
    else:
        #if the mouse is not over a scatter point
        ab.set_visible(False)
    fig.canvas.draw_idle()



def interactive_plot(image_ID, labels, x_coords, y_coords):
# =============================================================================
#     Inputs:
#     - ID is list of numbers corresponding to the number in the image file name.
#     - Labels is a list of cluster assignments corresponding to the image ID at the same index in ID.
#     - x_coords and y_coords are the coordinates from a scatter plot.
#     Output:
#     - interactive plot that displays the image when you hover over a point
#     (might be also able to define click/hover as a parameter using different motion notify events)
# =============================================================================
    
    # create figure and plot scatter
    global line, fig, ab, images, ID, x, y
    ID = image_ID
    x, y = x_coords, y_coords
       
    ## Add transparent points to plot for use by hover function:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line, = ax.plot(x,y, ls="", marker="o", markerfacecolor = "None", alpha = 0) # want this marker to be transparent, but still exist (if hover)
     
    ## Display color-coded t-SNE plot
    max_label = max(labels)
    min_label = min(labels)
    if min_label != max_label:
        num_clusters = max_label - min_label + 1 # assumes cluster labels are in 1-unit increments 
    else:
        num_clusters = 1
        
    
    plt.axis('equal')
    # design scatter plot with colormap mapping to the cluster labels (the array "labels")
    # Note that this plot will NOT be used by the hover function
    color_map = plt.get_cmap("viridis")
    
    if max_label != min_label: # more than a single cluster thus need colorbar
        norm = matplotlib.colors.BoundaryNorm(np.arange(min_label,num_clusters + 1,1), color_map.N)
        plt.scatter(x,y,s=25,c=labels, marker = 'o', cmap = color_map, norm = norm );
        plt.colorbar()
    else:
        plt.scatter(x,y,s=25,c=labels, marker = 'o', cmap = color_map );
    
    # add annotation box to the axes and make it invisible
    ax.add_artist(ab)
    ab.set_visible(False)
    
    # add callback for mouse moves
    fig.canvas.mpl_connect('motion_notify_event', hover)           
    plt.show() 
############## Ending code for interactive plot
