import matplotlib.pyplot as plt


def display_image(image):
    fig, ax = plt.subplots(1, figsize = (15,15))
    ax.imshow(image, cmap='gray', interpolation='nearest')
    ax.set_aspect('equal')
    plt.axis('off')
    plt.show()   