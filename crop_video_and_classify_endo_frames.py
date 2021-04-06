#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 19:22:01 2017

@author: felix & numan
"""

def fetch_largest_connected_component(binary):
    
    from skimage.measure import label, regionprops
    
    labelled = label(binary)
    uniq_labels = np.unique(labelled)
    areas = []

    regions = regionprops(labelled)
    
    for reg in regions:
        areas.append(reg.area)
        
    largest_region = uniq_labels[1:][np.argmax(areas)]

    return labelled == largest_region
    
    
def getVidRegion(inframe, bw_thresh=10):
    
    import cv2
    from skimage.filters import threshold_otsu
    frame_gray = cv2.cvtColor(inframe, cv2.COLOR_RGB2GRAY) # compute the luminance 

    # apply threshold: Otsu calculates optimal threshold to create binary image from grayscale image
    thresh = bw_thresh
    bw = frame_gray >= thresh
    
#    # double check 
#    plt.figure()    
#    plt.imshow(bw,cmap='gray')
#    plt.show()
    
    # fetch the largest connected component
    vidmask = fetch_largest_connected_component(bw)
    
    # fetch the bounding box coords.
    m, n = vidmask.shape
    X, Y = np.meshgrid(range(n), range(m))
    
    x_min = np.min(X[vidmask==1])
    x_max = np.max(X[vidmask==1])
    y_min = np.min(Y[vidmask==1])
    y_max = np.max(Y[vidmask==1])
    
    return vidmask, (x_min, x_max, y_min, y_max)
    
def draw_bounding_box(img, bbox,thickness):
    
    import cv2
    from skimage.util import img_as_ubyte
    
    (x_min, x_max, y_min, y_max) = bbox
    # we need the top-left corner () and bottom-right corner of the rectangle
    boxed_img = cv2.rectangle(img, (x_min,y_min),(x_max,y_max),(0,155,0),thickness)
    
    return boxed_img
    
def crop_video( img, mask_bbox):
    
    (x_min, x_max, y_min, y_max) = mask_bbox
    
    return img[y_min:y_max, x_min:x_max]


def classify_information(img, model, shape=(64,64)):
    
    from skimage.transform import resize
    #shape was 64x64, now changed it to 224x224
    im = resize(img, shape)
    
    info = model.predict(im[None,:])
    
    return info

def open_videoclip_moviepy(infile):
    
    from moviepy.editor import VideoFileClip
    
    clip = VideoFileClip(testvideofile)

     # extract the meta information from the file.
    clip_fps = clip.fps
    clip_duration = clip.duration
    n_frames = int(clip.duration/(1./clip_fps))

    return clip, [clip_fps, clip_duration, n_frames]
    
if __name__=="__main__":
    
    from moviepy.editor import *
    import pylab as plt 
    import numpy as np 
    from keras.models import load_model
    from skimage.transform import resize
    from tqdm import tqdm
    import gc
    
    
#==============================================================================
#   Load the pretrained CNN model (2-class binary classification)  
#==============================================================================
    #CNN_model_file = '../BN_CNN_network_64x64_scratch_informative_75split_less_stringent'
    #CNN_model_file = '../classify_densenet_190_0.7752382506926855_64_lr_updated.pth'
    CNN_model_file = './saved/CNN_network_128x128_positive_samples'
    CNN_classify_model = load_model(CNN_model_file)
    
    target_shape = (128,128)
    #target_shape = (192,108)
    #target_shape2 = (1079,1079)
    

    # give a test clip to try. 
    testvideofile = '../video_cleaning/M_28082019110119_0000000000004312_1_001_001-1.MP4'
    clip, (clip_fps, clip_duration, n_frames) = open_videoclip_moviepy(testvideofile)

#==============================================================================
#     Use a sample frame to crop the video.
#==============================================================================
    frame0 = clip.get_frame(0*clip_fps) # use the first frame to clip. 
    mask, mask_bbox = getVidRegion(frame0, bw_thresh=10) # get the crop % replace with Sharib's c code ??
    clipped = frame0[mask_bbox[2]:mask_bbox[3], mask_bbox[0]:mask_bbox[1]]
    
    
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.show()
    
    plt.figure()
    plt.imshow(draw_bounding_box(frame0, mask_bbox, thickness=10))
    plt.show()
    
    
##==============================================================================
##   Crop and classify information in endoscopy videos on the fly  
##==============================================================================

    sample_rate = 1
    batch_size = 1
    import cv2 as cv
    fourcc = cv.VideoWriter_fourcc(*'MPEG')
    out_pos2 = cv.VideoWriter('output_pos28_128.avi', fourcc, clip_fps, (1920, 1080))
    
    infoness = ['uninformative', 'informative']

    frame_scores = []

#    from scipy.misc import imsave

    n_batches = int(np.ceil(n_frames/float(batch_size)))


    for i in range(n_batches)[:]:

        if i <n_batches:
            start = i*batch_size
            end = (i+1)*batch_size
            #vid_frames2 = np.array([resize(crop_video( clip.get_frame(ii*1./clip_fps), mask_bbox ), target_shape2) for ii in range(start,end,1)]) 
            vid_frames = np.array([resize(crop_video( clip.get_frame(ii*1./clip_fps), mask_bbox ), target_shape) for ii in range(start,end,1)]) 
            #vid_frames = np.array([resize(clip.get_frame(ii*1./clip_fps), target_shape) for ii in range(start,end,1)]) 
            vid_frames2 = np.array([( clip.get_frame(ii*1./clip_fps) ) for ii in range(start,end,1)])
            
            # plt.figure()
            # plt.imshow(vid_frames[i,:,:,:], cmap='gray')
            # plt.show()
            
            # plt.figure()
            # plt.imshow(vid_frames2[i,:,:,:], cmap='gray')
            # plt.show()
    
        else:
            start = i*batch_size
            vid_frames = np.array([resize(crop_video( clip.get_frame(ii*1./clip_fps), mask_bbox ), target_shape) for ii in range(start,n_frames,1)])       

        informativeness = CNN_classify_model.predict(vid_frames)
        information_index = np.argmax(informativeness, axis=1)
        
        frame_scores.append(information_index)
        
        for j in range(vid_frames.shape[0]):
            if information_index[j] == 1:
        # writing to a image array
                #img_array = (vid_frames2[j,:,:,:] * 255).astype(np.uint8)
                img_array3 = (vid_frames2[j,:,:,:]).astype(np.uint8)
                img_array4 = cv.cvtColor(img_array3, cv.COLOR_RGB2BGR)
                out_pos2.write(img_array4)
                
        
        #aa=vid_frames[0,:,:,:]
        #out.write(vid_frames[0,:,:,:])
        #out.write(aa)
        
    frame_scores = np.hstack(frame_scores)
    
    
    fig, ax = plt.subplots()
    ax.imshow(frame_scores[None,:], cmap='coolwarm')
    ax.set_aspect('auto')
    plt.show()
    out_pos2.release()
    
# import cv2 as cv
# cap = cv.VideoCapture(0)
# # Define the codec and create VideoWriter object
# fourcc = cv.VideoWriter_fourcc(*'MJPG')
#fourcc = cv.VideoWriter_fourcc(*'FMP4')
#fourcc = cv2.VideoWriter_fourcc(*'MPEG')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
# out = cv.VideoWriter('output.MP4', fourcc, 50.0, (64,  64))
# out.write(frame_scores)

        
print("Done!!")
# plt.figure()
# plt.imshow(vid_frames2[90,:,:,:])
# plt.show()

# img_array = (vid_frames[0,:,:,:] * 255).astype(np.uint8)
