import streamlit as st
import cv2 
import numpy as np
from PIL import Image
import io
import pandas as pd
import argparse
from pyntcloud import PyntCloud   

path1=False
path2=False

def load_image(image_file):
	img = Image.open(image_file)
	return img

st.title("Point Cloud Formation ")

#st.sidebar.header("Demo")

st.subheader( "Choose Options")
config_select_options = st.selectbox("Select option:", ["Input files manually", "Input path"], 0)
if config_select_options == "Input files manually":
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    if image_file is not None:
        # To See details
	    file_details = {"filename":image_file.name, "filetype":image_file.type,"filesize":image_file.size}
        
# uploaded_file_doc = st.file_uploader(“Upload doc/docx Files”,type=[‘docx’,’doc’], accept_multiple_files=True)
else:

    input_path = st.text_input("Please input the path of your folder")
    uploaded_file = []
    if len(input_path) > 0:

        st.write(f"Stereo Image in this path {input_path} will be uploaded")
output_path = st.text_input("Please input the output path to house your Point Cloud")
if len(output_path) > 0:

    st.write(f"Amended Point Cloud will be housed in this path {output_path}")
st.write(" — — -")
#
#st.subheader("Choose type of manipulation to PDF")
#config_select_manipulation = st.multiselect("Select one or more options:", ["Add Watermark", "Remove Metadata", "Concatenate PDFs"], ["Add Watermark", "Remove Metadata", "Concatenate PDFs"])
#if "Add Watermark" in config_select_manipulation:

 #   uploaded_file_wmp = st.file_uploader("Upload watermark PDF for portrait",type=['pdf'])
 #   uploaded_file_wml = st.file_uploader("Upload watermark PDF for landscape",type=['pdf'])


#Main func
#lrsimilarity,tbsimilarity
def createdepthmap(left,right,img,lrsimilarity,tbsimilarity,fimage):
    print("Hi")
    depth_map = cv2.normalize(src=fimage, dst=fimage, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    depth_map = np.uint8(depth_map)
    
    if (tbsimilarity > lrsimilarity):
        depth_map = cv2.bitwise_not(depth_map)
    
    depth_image = Image.fromarray(depth_map, mode="L")
    colours_array  = np.array(left.resize(img.size)
                                .rotate(-90, expand=True)
                                .getdata()
                    ).reshape(img.size + (3,))
    return depth_image,colours_array

def lrmatcher(window_size,limage,rimage):
    print("Hi")
    lmatcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=16,
                blockSize=5,
                P1=8 * 3 * window_size ** 2,
                P2=32 * 3 * window_size ** 2,
            )         
    rmatcher = cv2.ximgproc.createRightMatcher(lmatcher)
    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=lmatcher)
    wls_filter.setLambda(80000)
    wls_filter.setSigmaColor(1.2)
    left_disparity  = lmatcher.compute(limage, rimage)
    right_disparity = rmatcher.compute(rimage, limage)
    left_disparity  = np.int16(left_disparity)
    right_disparity = np.int16(right_disparity)
    imagefiltered  = wls_filter.filter(left_disparity, limage, None, right_disparity)
    return imagefiltered

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    print(err)
    return err

def createmesh(image_file):
    img_0=load_image(image_file)
    img= img_0.convert('RGB')
    print("Hi")
    w, h = img.size
    r  = img.crop( (0,0, w/2, h))
    l   = img.crop( (w/2, 0, w,h))
    t    = img.crop( (0,0, w, h/2))
    b = img.crop( (0, h/2, w,h))
    
    try:
        l_r_same = mse(np.array(r),np.array(l)) 
        print(l_r_same)

        t_b_same = mse(np.array(t),np.array(b)) 
        print(t_b_same)
     
    except:
        st.warning("The image is not Stereoscopic!")
        return 0
    
    if (t_b_same < l_r_same):
        l  = b
        r = t
    

    image_l  = np.array(l) 
    image_r = np.array(r) 
    window_size = 15
    filtered_image  =lrmatcher(window_size,image_l,image_r)
    #,l_r_same,t_b_same,
    depth_image,coloursarray=createdepthmap(l,r,img,l_r_same, t_b_same,filtered_image)
    indicesarray = np.moveaxis(np.indices(img.size), 0, 2)
    image_Array    = np.dstack((indicesarray, coloursarray)).reshape((-1,5))
    df = pd.DataFrame(image_Array, columns=["x", "y", "red","green","blue"])
    depths_array = np.array(depth_image.resize(img.size)
                                        .rotate(-90, expand=True)
                                        .getdata())     
    df.insert(loc=2, column='z', value=depths_array)
    df[['red','green','blue']] = df[['red','green','blue']].astype(np.uint)
    df[['x','y','z']] = df[['x','y','z']].astype(float)
    df['z'] = df['z']*5
    cloud = PyntCloud(df)
    print("OK")
    cloud.to_file(output_path+"output"+".ply", also_save=["mesh","points"],as_text=True)
    print("Done")
    return 1





if st.button("Createmesh"):
    #st.image(uploaded_file)
    st.write(file_details)
    # To View Uploaded Image
    st.image(load_image(image_file),width=250)
    createmesh(image_file)
    