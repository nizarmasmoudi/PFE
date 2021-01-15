def show_img(img_path, draw_bbox=False, figsize=(20, 20)):
    '''
        Arguments :
            - img_path : Path leading the image
            - draw_bbox (False) : Set True to draw bounding boxes using annotations
            - figsize : Size of matplotlib figure
        Note : This function is specific to VisDrone Dataset. It will most likely not work with other datasets
    '''
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import pandas as pd

    # Showing raw image
    image = plt.imread(img_path)
    plt.figure(figsize=figsize)
    plt.imshow(image)
    # Drawing bounding boxes
    if draw_bbox:
        annotations = pd.read_csv(img_path.replace('images', 'annotations').replace('.jpg', '.txt'), header = None)
        annotations = annotations[annotations[5].isin([1, 2])]
        annotations = annotations.reset_index().drop(columns = ['index'])
        for i in range(len(annotations)):
            annotation0 = annotations.loc[i].values
            box = patches.Rectangle(xy = (annotation0[0], annotation0[1]), 
                                    width = annotation0[2], 
                                    height = annotation0[3], 
                                    linewidth = 1, 
                                    edgecolor = 'r', 
                                    facecolor = 'none')
            plt.gca().add_patch(box)
    # Cleaning matplotlib figure
    plt.tick_params(which='both', 
                    bottom = False, top = False, left = False, right = False,
                    labelbottom = False, labeltop = False, labelleft = False, labelright = False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    plt.show()