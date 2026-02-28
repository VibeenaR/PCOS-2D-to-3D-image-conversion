# import torch
# import cv2
# import numpy as np
# from torchvision import transforms, models
# import torch.nn as nn
# import open3d as o3d
# import os
# import config
# import shutil

# # --------------------------
# # Paths
# # --------------------------
# test_folder = r"C:/Users/vibee/vspcos/test_images.png"  
# # Clear previous outputs
# if os.path.exists(config.OUTPUT_DIR):
#     shutil.rmtree(config.OUTPUT_DIR)
# os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# # --------------------------
# # Load trained model
# # --------------------------
# model = models.resnet18()
# model.fc = nn.Linear(model.fc.in_features, 2)  # normal / pcos

# try:
#     state_dict = torch.load(config.MODEL_PATH, map_location=config.DEVICE, weights_only=True)
#     model.load_state_dict(state_dict)
# except TypeError:
#     model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))

# model.eval()
# model.to(config.DEVICE)

# # --------------------------
# # Image transform
# # --------------------------
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
#     transforms.ToTensor(),
# ])

# # --------------------------
# # Collect all images
# # --------------------------
# if not os.path.exists(test_folder):
#     print("ERROR: Test folder does not exist:", test_folder)
#     exit()

# images = [os.path.join(test_folder, f) for f in sorted(os.listdir(test_folder))
#           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# if len(images) == 0:
#     print("No images found in test folder:", test_folder)
#     exit()

# print(f"Processing {len(images)} independent images...")

# # --------------------------
# # Process each image individually
# # --------------------------
# for idx, image_path in enumerate(images):
#     img_file = os.path.basename(image_path)
#     print(f"\nProcessing Image {idx+1}: {img_file}")

#     # Load image
#     image = cv2.imread(image_path)
#     if image is None:
#         print("ERROR: Failed to read image:", image_path)
#         continue

#     # Prediction
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     input_tensor = transform(image_rgb).unsqueeze(0).to(config.DEVICE)

#     with torch.no_grad():
#         output = model(input_tensor)
#         predicted = torch.argmax(output, 1)
#     print("Prediction:", config.CLASSES[predicted.item()])

#     # Depth map
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Smooth depth map to reduce noise
#     gray_smooth = cv2.GaussianBlur(gray, (5, 5), 0)

#     depth = cv2.normalize(gray_smooth, None, 0, 255, cv2.NORM_MINMAX)

#     # Save depth map
#     depth_filename = os.path.join(config.OUTPUT_DIR, f"depth_map_{img_file}")
#     cv2.imwrite(depth_filename, depth)
#     print("Depth map saved at:", depth_filename)

#     # Generate 3D point cloud
#     h, w = depth.shape
#     xs, ys = np.meshgrid(np.arange(w), np.arange(h))

#     # Exaggerate Z-axis for visualization
#     zs = depth / 15.0  # adjust divisor to make height taller or shorter

#     points = np.stack([xs.flatten(), ys.flatten(), zs.flatten()], axis=1)

#     # Apply bright JET colormap
#     colored_image = cv2.applyColorMap(gray_smooth, cv2.COLORMAP_JET)
#     colors = cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)

#     # Smooth point cloud using voxel downsample
#     pcd = pcd.voxel_down_sample(voxel_size=1.0)

#     # Estimate normals for visualization
#     pcd.estimate_normals()

#     # Optional: Taubin smoothing (if Open3D version supports it)
#     try:
#         pcd = pcd.filter_smooth_taubin(number_of_iterations=5)
#     except AttributeError:
#         pass  # skip if not available

#     # Save point cloud
#     pcd_filename = os.path.join(config.OUTPUT_DIR, f"point_cloud_{img_file}.ply")
#     o3d.io.write_point_cloud(pcd_filename, pcd)
#     print("3D point cloud saved at:", pcd_filename)

# print("\nAll images processed! You now have individual depth maps and 3D point clouds for each ultrasound image.")

import torch
import cv2
import numpy as np
from torchvision import transforms, models
import torch.nn as nn
import open3d as o3d
import os
import config
import shutil

# --------------------------
# Paths
# --------------------------
test_folder = r"C:/Users/vibee/vspcos/test_images.png"  
# Clear previous outputs
if os.path.exists(config.OUTPUT_DIR):
    shutil.rmtree(config.OUTPUT_DIR)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

# --------------------------
# Load trained model
# --------------------------
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, 2)  # normal / pcos

try:
    state_dict = torch.load(config.MODEL_PATH, map_location=config.DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
except TypeError:
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))

model.eval()
model.to(config.DEVICE)

# --------------------------
# Image transform
# --------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
])

# --------------------------
# Collect all images
# --------------------------
if not os.path.exists(test_folder):
    print("ERROR: Test folder does not exist:", test_folder)
    exit()

images = [os.path.join(test_folder, f) for f in sorted(os.listdir(test_folder))
          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if len(images) == 0:
    print("No images found in test folder:", test_folder)
    exit()

print(f"Processing {len(images)} independent images...")

# --------------------------
# Process each image individually
# --------------------------
for idx, image_path in enumerate(images):
    img_file = os.path.basename(image_path)
    print(f"\nProcessing Image {idx+1}: {img_file}")

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("ERROR: Failed to read image:", image_path)
        continue

    # Prediction
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image_rgb).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        output = model(input_tensor)
        predicted = torch.argmax(output, 1)
    print("Prediction:", config.CLASSES[predicted.item()])

    # Depth map
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Smooth depth map to reduce noise
    gray_smooth = cv2.GaussianBlur(gray, (5, 5), 0)

    depth = cv2.normalize(gray_smooth, None, 0, 255, cv2.NORM_MINMAX)

    # Save depth map
    depth_filename = os.path.join(config.OUTPUT_DIR, f"depth_map_{img_file}")
    cv2.imwrite(depth_filename, depth)
    print("Depth map saved at:", depth_filename)

    # Generate 3D point cloud
    h, w = depth.shape
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))

    # Exaggerate Z-axis for visualization
    zs = depth / 15.0  # adjust divisor to make height taller or shorter

    points = np.stack([xs.flatten(), ys.flatten(), zs.flatten()], axis=1)

    # Apply bright JET colormap
    colored_image = cv2.applyColorMap(gray_smooth, cv2.COLORMAP_JET)
    colors = cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Smooth point cloud using voxel downsample
    pcd = pcd.voxel_down_sample(voxel_size=1.0)

    # Estimate normals for visualization
    pcd.estimate_normals()

    # Optional: Taubin smoothing (if Open3D version supports it)
    try:
        pcd = pcd.filter_smooth_taubin(number_of_iterations=5)
    except AttributeError:
        pass  # skip if not available

    # Save point cloud
    pcd_filename = os.path.join(config.OUTPUT_DIR, f"point_cloud_{img_file}.ply")
    o3d.io.write_point_cloud(pcd_filename, pcd)
    print("3D point cloud saved at:", pcd_filename)

print("\nAll images processed! You now have individual depth maps and 3D point clouds for each ultrasound image.")