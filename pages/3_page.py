

import streamlit as st
import numpy as np
import open3d as o3d
import io


st.title("Point cloud Filtering")

#st.sidebar.header("Plotting Demo")

from PIL import Image


#cloud_file = st.file_uploader("Upload point cloud", type=["ply","xyz"])
input_path = st.text_input("Please input the path of your folder")
uploaded_file = []
if len(input_path) > 0:
    
    if input_path.endswith(('.xyz','.ply')):
        st.write(f"Point Cloud in this path {input_path} will be uploaded")
    else:
        st.write(f"Point cloud is not of ply , xyz format.")
        input_path=' '

output_path = st.text_input("Please input the output path to house your Outputs")
if len(output_path) > 0:

    st.write(f"Outputs will be housed in this path {output_path}")
st.write(" — — -")

def load_cloud(cloud_file):
	pcd = o3d.io.read_point_cloud(cloud_file)
	return pcd

def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    pcd_o=outlier_cloud.paint_uniform_color([1, 0, 0])
    pcd_i=inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    #o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
    merged_pcd=pcd_o+pcd_i
    o3d.io.write_point_cloud("Output_Outlier.ply",merged_pcd )
	

def voxel(input_path):
	print("Downsample the point cloud with a voxel of 0.02")
    
	pcd=o3d.io.read_point_cloud(input_path)
    
	voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
	o3d.io.write_point_cloud("Voxel_output.ply",voxel_down_pcd )

def uniform(pcd):
    print("Every 5th points are selected")
    pcd=o3d.io.read_point_cloud(input_path)
    uni_down_pcd = pcd.uniform_down_sample(every_k_points=5)
    #o3d.visualization.draw_geometries([uni_down_pcd])
    o3d.io.write_point_cloud("Uniform_output.ply",uni_down_pcd )

def statistical(pcd):
    print("Statistical oulier removal")

    #cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
    #                                                   std_ratio=2.0)
    pcd=o3d.io.read_point_cloud(input_path)
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
    cl, ind = voxel_down_pcd.remove_statistical_outlier(20, 2.0)
    #down_pcd = down_pcd.select_down_sample(ind)
    display_inlier_outlier(voxel_down_pcd, ind)

def radius(pcd):    
    print("Radius oulier removal")
    pcd=o3d.io.read_point_cloud(input_path)
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
    cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    display_inlier_outlier(voxel_down_pcd, ind)	


st.header("Voxel Filtering")
st.write("This method reduces the number of points in the point cloud by voxelizing the point cloud into a 3D grid and taking one point per grid cell. It is a fast and effective way to reduce the size of large point clouds while retaining their overall structure")
number = st.number_input('Enter the voxel Size: ')
if st.button("Voxel"):
    #st.image(uploaded_file)
    #st.write(file_details)
    #pcd=load_cloud(input_path)
    
    voxel(input_path)


st.header("Uniform Filtering")
st.write("This method divides the point cloud into a grid of uniform cells and then choose one representative point from each cell.")
number = st.number_input('Enter the no of points: ',min_value=0,max_value=100,key="00")
if st.button("Uniform"):
    #st.image(uploaded_file)
    #st.write(file_details)
    #pcd=load_cloud(input_path)
    uniform(input_path)

st.header("Statstical oulier removal")

st.write("This method removes points that are significantly different from their neighbors based on statistical measures. It is often used to remove noisy points from the point cloud.")

col1,col2 = st.columns(2)

number = st.number_input('Enter the Voxel Size : ',min_value=0.00,max_value=100.0,key="1")

with col1:
    #st.success("From Col1")
    number1 = st.number_input('Enter the Neighbour : ',min_value=0,max_value=100,key="2")
with col2:
   #st.info("From Col2")
   number2 = st.number_input('Enter the Standard ratio : ',min_value=0.00,max_value=100.0,key="3")

if st.button("Statstical"):
    #st.image(uploaded_file)
    #st.write(file_details)
    #pcd=load_cloud(input_path)
    
    statistical(input_path)

st.header("Radius Outlier Removal")

st.write("This method removes points that have fewer than a specified number of neighbors within a specified radius. It is often used to remove isolated points from the point cloud.")

col3,col4 = st.columns(2)
with col3:
    #st.success("From Col1")
    number1 = st.number_input('Enter the Neighbour : ',min_value=0,max_value=100,key="4")
with col4:
   #st.info("From Col2")
   number2 = st.number_input('Enter the Radius : ',min_value=0.00,max_value=100.0,key="5")


if st.button("Radius"):
    #st.image(uploaded_file)
    #st.write(file_details)
    #pcd=load_cloud(input_path)
    voxel_down_pcd=voxel(input_path)
    radius(voxel_down_pcd)