import open3d as o3d

# Load LiDAR point cloud data (replace 'your_point_cloud.pcd' with the actual file path)
point_cloud = o3d.io.read_point_cloud('your_point_cloud.pcd')

# Downsample the point cloud
voxel_size = 0.05
downsampled_cloud = point_cloud.voxel_down_sample(voxel_size)

# Remove ground points (replace 'ground_threshold' with the actual threshold)
ground_threshold = 0.2
non_ground_cloud = downsampled_cloud.select_by_function(
    lambda p: p[2] > ground_threshold
)

# Cluster the remaining points
clustering_distance = 0.1
labels = non_ground_cloud.cluster_dbscan(eps=clustering_distance, min_points=10)

# Visualize the original point cloud and clusters
o3d.visualization.draw_geometries([point_cloud, non_ground_cloud])

# Print the number of clusters
num_clusters = len(set(labels))
print(f'Number of clusters: {num_clusters}')