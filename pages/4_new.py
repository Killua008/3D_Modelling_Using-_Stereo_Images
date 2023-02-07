"""
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Load the point cloud
pcd = o3d.io.read_point_cloud("Dial_M_For_Murder_02.png.ply")

# Convert point cloud to numpy array
points = np.asarray(pcd.points)

# Perform DBSCAN clustering
model_dbscan = DBSCAN(eps=0.05, min_samples=10)
labels = model_dbscan.fit_predict(points)

# Create a point cloud for each cluster
clusters = []
for label in np.unique(labels):
    cluster = points[labels==label]
    clusters.append(cluster)

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

# Visualize the point cloud
#colors = np.array([[1, 0, 0] if label == 0 else [0, 1, 0] for label in labels])
#pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud("copy_of_fragmentco.ply",pcd )
#o3d.visualization.draw_geometries([pcd])
"""

#V2
"""
import open3d as o3d

# Load the point cloud from a file
pcd = o3d.io.read_point_cloud("Dial_M_For_Murder_02.png.ply")



# Create a VoxelGrid filter
voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.01)

# Create a PassThrough filter
pass_through_filter = o3d.geometry.PassThroughFilter()
pass_through_filter.set_bounding_box(min_bound = [-1, -1, -1], max_bound = [1, 1, 1])
pass_through_filter.set_filter_field_name("z")
pass_through_filter.set_filter_limits(min_bound = -1, max_bound = 1)

# Apply the filter
cloud_filtered = pass_through_filter.filter(voxel_down_pcd)

# Save the filtered point cloud to a file
o3d.io.write_point_cloud("output.ply", cloud_filtered)
"""

#Ransac and Dbscan segmentation

"""
import streamlit as st
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import io


st.title("Second Page")

st.sidebar.header("Plotting Demo")

from PIL import Image





cloud_file = st.file_uploader("Upload point cloud", type=["ply","xyz"])
input_path = st.text_input("Please input the path of your folder")
uploaded_file = []
if len(input_path) > 0:

    st.write(f"PDFs in this path {input_path} will be uploaded")
"""
"""
if cloud_file is not None:

		# To See details
		file_details = {"filename":cloud_file.name, "filetype":cloud_file.type,
                              "filesize":cloud_file.size}
		st.write(file_details)

        # To View Uploaded Image_filecloud_file
"""
"""
output_path = st.text_input("Please input the output path to house your Outputs")
if len(output_path) > 0:

    st.write(f"Outputs will be housed in this path {output_path}")
st.write(" — — -")

def load_cloud(cloud_file):
	
	
	pcd = o3d.io.read_point_cloud(cloud_file)
	return pcd

def pcd1(cloud_file):
	pcd=load_cloud(cloud_file)
	pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=16), fast_normal_computation=True)
	pcd.paint_uniform_color([0.6, 0.6, 0.6])
#o3d.visualization.draw_geometries([pcd]) #Works only outside Jupyter/Colab

## 3.2 [INITIATION] 3D Shape Detection with RANSAC

	try:
		plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,ransac_n=3,num_iterations=1000)
	except:
		st.warning("The Point cloud must have atleast 3 points to obtain Inliners!!")
		return 0
	[a, b, c, d] = plane_model
	print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
	inlier_cloud = pcd.select_by_index(inliers)
	outlier_cloud = pcd.select_by_index(inliers, invert=True)
	inlier_cloud.paint_uniform_color([1.0, 0, 0])
	outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
	

	labels = np.array(pcd.cluster_dbscan(eps=0.05, min_points=10))
	max_label = labels.max()
	print(f"point cloud has {max_label + 1} clusters")

	colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
	colors[labels < 0] = 0
	pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])


	segment_models={}
	segments={}
	max_plane_idx=10
	rest=pcd
	for i in range(max_plane_idx):
		colors=plt.get_cmap("tab20")(i)
		segment_models[i], inliers = rest.segment_plane(distance_threshold=0.01,ransac_n=3,num_iterations=1000)
		segments[i]=rest.select_by_index(inliers)
		segments[i].paint_uniform_color(list(colors[:3]))
		rest = rest.select_by_index(inliers, invert=True)
		print("pass",i,"/",max_plane_idx,"done.")

    ## 4.2 Refined RANSAC with Euclidean clustering

	segment_models={}
	segments={}
	max_plane_idx=20
	rest=pcd
	d_threshold=0.01
	for i in range(max_plane_idx):
		colors = plt.get_cmap("tab20")(i)
		segment_models[i], inliers = rest.segment_plane(distance_threshold=0.01,ransac_n=3,num_iterations=1000)
		segments[i]=rest.select_by_index(inliers)
		labels = np.array(segments[i].cluster_dbscan(eps=d_threshold*10, min_points=10))
		candidates=[len(np.where(labels==j)[0]) for j in np.unique(labels)]
		best_candidate=int(np.unique(labels)[np.where(candidates==np.max(candidates))[0]])
		print("the best candidate is: ", best_candidate)
		rest = rest.select_by_index(inliers, invert=True)+segments[i].select_by_index(list(np.where(labels!=best_candidate)[0]))
		segments[i]=segments[i].select_by_index(list(np.where(labels==best_candidate)[0]))
		segments[i].paint_uniform_color(list(colors[:3]))
		print("pass",i+1,"/",max_plane_idx,"done.")	


	## 4.3 Euclidean clustering of the rest with DBSCAN

	labels = np.array(rest.cluster_dbscan(eps=0.05, min_points=5))
	max_label = labels.max()
	print(f"point cloud has {max_label + 1} clusters")

	colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
	colors[labels < 0] = 0
	rest.colors = o3d.utility.Vector3dVector(colors[:, :3])

# o3d.visualization.draw_geometries([segments.values()])
# o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest])
#o3d.visualization.draw_geometries([segments[i] for i in range(max_plane_idx)]+[rest],zoom=0.3199,front=[0.30159062875123849, 0.94077325609922868, 0.15488309545553303],lookat=[-3.9559999108314514, -0.055000066757202148, -0.27599999308586121],up=[-0.044411423633999815, -0.138726419067636, 0.98753122516983349])
# o3d.visualization.draw_geometries([rest])

	o3d.io.write_point_cloud(output_path+"copy_of_fragment.ply", rest )
	st.write("Done your output is saved.")


if st.button("Segment"):
    #st.image(uploaded_file)
    #st.write(file_details)
    
    pcd1(input_path)
"""
"""
import open3d as o3d


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud])
    #o3d.io.write_point_cloud("output00.ply",[inlier_cloud, outlier_cloud] )


if __name__ == "__main__":

    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud("Planet_Dinosaur_1.png.ply")
    o3d.visualization.draw_geometries([pcd])

    print("Downsample the point cloud with a voxel of 0.02")
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
    o3d.visualization.draw_geometries([voxel_down_pcd])
    #o3d.io.write_point_cloud("Voxel_output.ply",voxel_down_pcd )

    print("Every 5th points are selected")
    uni_down_pcd = pcd.uniform_down_sample(every_k_points=5)
    o3d.visualization.draw_geometries([uni_down_pcd])
    #o3d.io.write_point_cloud("Uniform_output.ply",uni_down_pcd )
    
    print("Statistical oulier removal")
    #cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20,
    #                                                   std_ratio=2.0)
    cl, ind = voxel_down_pcd.remove_statistical_outlier(20, 2.0)
    #down_pcd = down_pcd.select_down_sample(ind)
    display_inlier_outlier(voxel_down_pcd, ind)
    
    print("Radius oulier removal")
    cl, ind = voxel_down_pcd.remove_radius_outlier(nb_points=16, radius=0.05)
    display_inlier_outlier(voxel_down_pcd, ind)
"""
	
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Load the point cloud
pcd = o3d.io.read_point_cloud("the_researcher_desk.xyz")

# Convert point cloud to numpy array
points = np.asarray(pcd.points)

# Perform DBSCAN clustering
model_dbscan = DBSCAN(eps=0.05, min_samples=10)
labels = model_dbscan.fit_predict(points)

# Create a point cloud for each cluster
clusters = []
for label in np.unique(labels):
    cluster = points[labels==label]
    clusters.append(cluster)

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")

colors = plt.get_cmap("tab10")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

# Visualize the point cloud
#colors = np.array([[1, 0, 0] if label == 0 else [0, 1, 0] for label in labels])
#pcd.colors = o3d.utility.Vector3dVector(colors)
#o3d.io.write_point_cloud("copy_of_fragmentco.ply",pcd )
o3d.visualization.draw_geometries([pcd])