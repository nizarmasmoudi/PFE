from processing.image import Image, read_image
import dataset.navigation as nav
import pandas as pd
import os

img_path = os.path.join(nav.VALID_PATH, 'images', '0000333_01765_d_0000010.jpg')
image = read_image(img_path)
image.pad()

h_overlap, v_overlap = 736*2 - image.width, 736*2 - image.height

crops = [r'trash\images\0000333_01765_d_0000010_0.jpg', r'trash\images\0000333_01765_d_0000010_2.jpg', r'trash\images\0000333_01765_d_0000010_1.jpg', r'trash\images\0000333_01765_d_0000010_3.jpg']
crops = [read_image(crop) for crop in sorted(crops)]

if v_overlap == 736:
    df0, df2 = [crop.annotation for crop in crops]
    df2['bbox_left'] = df2['bbox_left'] + (736 - h_overlap)
    df = pd.concat((
        df0.loc[df0['bbox_left'] < 736 - h_overlap, :], df2.loc[df2['bbox_left'] > 736]
    ), axis = 0)
    df0 = df0.loc[df0['bbox_left'] >= 736 - h_overlap, :]
    df2 = df2.loc[df2['bbox_left'] <= 736]
    df = pd.concat((
        df, df0 if len(df0) > len(df2) else df2 #! Change condition here
    ), axis = 0)
elif h_overlap == 736:
    df0, df1 = [crop.annotation for crop in crops]
    df1['bbox_top'] = df1['bbox_top'] + (736 - v_overlap)
    df = pd.concat((
        df0.loc[df0['bbox_top'] < 736 - v_overlap, :], df1.loc[df1['bbox_top'] > 736]
    ), axis = 0)
    df0 = df0.loc[df0['bbox_top'] >= 736 - v_overlap, :]
    df1 = df1.loc[df1['bbox_top'] <= 736]
    df = pd.concat((
        df, df0 if len(df0) > len(df1) else df1 #! Change condition here
    ), axis = 0)
else:
    df0, df1, df2, df3 = [crop.annotation for crop in crops]
    df1['bbox_top'] = df1['bbox_top'] + (736 - v_overlap)
    df2['bbox_left'] = df2['bbox_left'] + (736 - h_overlap)
    df3['bbox_top'] = df3['bbox_top'] + (736 - v_overlap)
    df3['bbox_left'] = df3['bbox_left'] + (736 - h_overlap)
    df = pd.concat((
        df0.loc[(df0['bbox_left'] < 736 - h_overlap) & (df0['bbox_top'] < 736 - v_overlap), :],
        df1.loc[(df1['bbox_left'] < 736 - h_overlap) & (df1['bbox_top'] > 736), :],
        df2.loc[(df2['bbox_left'] > 736) & (df2['bbox_top'] < 736 - v_overlap), :],
        df3.loc[(df3['bbox_left'] > 736) & (df3['bbox_top'] > 736), :]
    ), axis = 0)
    # Green
    ovp_01 = df0.loc[(df0['bbox_left'] < 736 - h_overlap) & (df0['bbox_top'] >= 736 - v_overlap), :]
    ovp_10 = df1.loc[(df1['bbox_left'] < 736 - h_overlap) & (df1['bbox_top'] <= 736), :]
    # Orange
    ovp_02 = df0.loc[(df0['bbox_left'] >= 736 - h_overlap) & (df0['bbox_top'] < 736 - v_overlap), :]
    ovp_20 = df2.loc[(df2['bbox_left'] <= 736) & (df2['bbox_top'] < 736 - v_overlap), :]
    # Yellow
    ovp_23 = df2.loc[(df2['bbox_left'] > 736) & (df2['bbox_top'] >= 736 - v_overlap), :]
    ovp_32 = df3.loc[(df3['bbox_left'] > 736) & (df3['bbox_top'] <= 736), :]
    # Blue
    ovp_13 = df1.loc[(df1['bbox_left'] >= 736 - h_overlap) & (df1['bbox_top'] > 736), :]
    ovp_31 = df3.loc[(df3['bbox_left'] <= 736) & (df3['bbox_top'] > 736), :]
    # Red
    sqr_0 = df0.loc[(df0['bbox_left'] >= 736 - h_overlap) & (df0['bbox_top'] >= 736 - v_overlap), :]
    sqr_1 = df1.loc[(df1['bbox_left'] >= 736 - h_overlap) & (df1['bbox_top'] <= 736), :]
    sqr_2 = df2.loc[(df2['bbox_left'] <= 736) & (df2['bbox_top'] >= 736 - v_overlap), :]
    sqr_3 = df3.loc[(df3['bbox_left'] <= 736) & (df3['bbox_top'] <= 736), :]
    df = pd.concat((
        df,
        ovp_01 if len(ovp_01) > len(ovp_10) else ovp_10, #! Change conditon here
        ovp_02 if len(ovp_02) > len(ovp_20) else ovp_20, #! Change conditon here
        ovp_23 if len(ovp_23) > len(ovp_32) else ovp_32, #! Change conditon here
        ovp_13 if len(ovp_13) > len(ovp_31) else ovp_31, #! Change conditon here
        sorted([sqr_0, sqr_1, sqr_2, sqr_3], key = lambda sqr: len(sqr))[-1]
    ), axis = 0)
    
    
print('Original annotation:')
print(df)
print('Collected annotation:')
print(image.annotation)

test = Image(image.mat, df, title=' ')
test.display(True)