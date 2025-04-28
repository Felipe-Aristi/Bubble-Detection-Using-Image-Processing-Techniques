import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, segmentation
from skimage.measure import label, regionprops
from skimage.segmentation import clear_border 





image_path = r"C:\Users\juanp\Desktop\Research project\TEST3\0.8\Images\2000_0.8_368.jpg"


img = cv2.imread(image_path)

img_B, img_G, img_R = cv2.split(img)

img2 = cv2.merge((img_R, img_G, img_B))
sys.getsizeof(img2)
plt.imshow(img2), plt.grid('off'), plt.xticks([]), plt.yticks([]), plt.title('Original RGB Image')

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1), plt.imshow(img_B, cmap='gray'), plt.title('Blue Channel'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 2), plt.imshow(img_G, cmap='gray'), plt.title('Green Channel'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 3), plt.imshow(img_R, cmap='gray'), plt.title('Red Channel'), plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 4), plt.imshow(img_B, cmap='jet'), plt.title('Blue Channel (Jet)'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 5), plt.imshow(img_G, cmap='jet'), plt.title('Green Channel (Jet)'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 6), plt.imshow(img_R, cmap='jet'), plt.title('Red Channel (Jet)'), plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()


# Binarize the image using adaptive thresholding on the blue channel
binImg = cv2.adaptiveThreshold(img_G, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 15)

dst_inv = cv2.bitwise_not(binImg)
plt.figure()
plt.title("Binarized Image")
plt.imshow(dst_inv, cmap='gray')
plt.axis('off')
plt.show()

img_clb = clear_border(dst_inv, buffer_size=3, bgval=1)
"""
plt.figure()
plt.title("Image with Cleared Border")
plt.imshow(img_clb, cmap='gray')
plt.axis('off')
plt.show()
"""

"---------------------------------First Filter--------------------------------------------------------"
labeled_img = label(img_clb)

y_point = 600
x_point = 200

# Create a mask to store the filtered result
filtered_mask = np.zeros_like(img_clb)

y_centroids = []
x_centroids = []


plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(img_clb, cv2.COLOR_BGR2RGB))

# Get the maximum x-coordinate (image width)
image_width = labeled_img.shape[1]

for i, region in enumerate(regionprops(labeled_img)):
    y, x = region.centroid

    # Modify x-coordinate to store the "flipped" x-value
    flipped_x = image_width - x

    # Append centroids to their respective lists
    y_centroids.append(y)
    x_centroids.append(flipped_x)
    
    # Draw the label number at the centroid
    plt.text(x, y, str(i + 1), color='white', fontsize=4, ha='center', va='center')

plt.title('Labeled Regions with Numbers (Before Filtering)')
plt.axis('off')
plt.show()


# Filter objects based on both x and y centroids
filtered_mask = np.zeros_like(labeled_img, dtype=np.uint8)

for region in regionprops(labeled_img):
    y_centroid = region.centroid[0]
    x_centroid = region.centroid[1]
    
    # Filter objects based on both thresholds
    if y_centroid <= y_point and x_centroid > x_point:
        filtered_mask[labeled_img == region.label] = 1


# Convert the filtered mask to the appropriate data type
filtered_mask = (filtered_mask * 255).astype(np.uint8)

# Apply the filtered mask to the clean image
img_clb_filtered = cv2.bitwise_and(img_clb, filtered_mask)

# Apply the filtered mask to the original image while keeping the background intact
masked_rgb = cv2.bitwise_and(img, img, mask=filtered_mask)

# Ensure that the background is white instead of black
white_background = np.full_like(img, 255)  # Create a white background
img_outl = np.where(filtered_mask[..., None] == 255, masked_rgb, white_background)  # Apply mask

# Display the corrected filtered image
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(img_outl, cv2.COLOR_BGR2RGB))
plt.title('Filtered Objects with Preserved Background')
plt.axis('off')
plt.show()

"-----------------------------------------------------------------------------------------------------"

# Apply Gaussian Blur to Reduce Noise
blurred_image = cv2.GaussianBlur(img_clb_filtered, (3, 3), 0)

# Otsu’s Threshold to Determine an Optimal Threshold
otsu_threshold, thresholded_img = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#Canny Edge Detection with Dynamic Thresholds
edges = cv2.Canny(blurred_image, otsu_threshold * 0.5, otsu_threshold * 1.5)

plt.figure(figsize=(8, 8))
plt.imshow(edges, cmap='gray')
plt.title('Detected Edges')
plt.axis('off')
plt.show()

# Dilation
kernel = np.ones((2, 2), np.uint8)
dst = cv2.dilate(edges, kernel, iterations=1)

plt.figure(figsize=(8, 8))
plt.imshow(dst, cmap='gray')
plt.title('Dilated Image')
plt.axis('off')
plt.show()

# Fill the closed objects in the dilation
h, w = dst.shape[:2]
mask = np.zeros((h + 2, w + 2), np.uint8)
flooded = dst.copy()
cv2.floodFill(flooded, mask, (0, 0), 255)

# Invert the filled image
flood_inv = cv2.bitwise_not(flooded)

# Combine the images to get the background
im_out = dst | flood_inv

plt.figure(figsize=(8, 8))
plt.imshow(im_out, cmap='gray')
plt.title('Filled Objects')
plt.axis('off')
plt.show()

"Apply additional morphological operations"
kernel = np.ones((2, 2), np.uint8)
imgclose = cv2.morphologyEx(im_out, cv2.MORPH_CLOSE, kernel)
imgclgr = cv2.morphologyEx(imgclose, cv2.MORPH_GRADIENT, kernel)

plt.figure(figsize=(8, 8))
plt.imshow(imgclgr, cmap='gray')
plt.title('Morphological Gradient')
plt.axis('off')
plt.show()

# Flood Fill Function
h, w = imgclgr.shape[:2]
flood_mask = np.zeros((h + 2, w + 2), np.uint8)  # Create a mask for flood filling

flood_filled_image = imgclgr.copy()
cv2.floodFill(flood_filled_image, flood_mask, (0, 0), 255)
flood_inv = cv2.bitwise_not(flood_filled_image)
final_filled_mask = (flood_inv > 0).astype(np.uint8) * 255


final_result = np.where(final_filled_mask[..., None] == 255, 
                        np.array([255, 255, 255], dtype=np.uint8), 
                        img)

plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
plt.title('Final Recognized Bubbles')
plt.axis('off')
plt.show()


"---------------Second filter-----------------------------------------------------------------------------"
C_Threshold = 0.8  # Circularity threshold

# Label the connected components in the final filled mask
labeled_bubbles = label(final_filled_mask > 0)
props = regionprops(labeled_bubbles)

# Count total bubbles and total area before filtering
total_bubbles = len(props)
total_area = sum(region.area for region in props)

# Create an empty mask to store only the acceptable (circular) bubbles
filtered_bubbles_mask = np.zeros_like(final_filled_mask, dtype=np.uint8)

# Initialize counters for remaining bubbles and remaining area
remaining_bubbles = 0
remaining_area = 0

# Process each detected bubble
for region in props:
    area = region.area
    perimeter = region.perimeter
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    if circularity >= C_Threshold:
        filtered_bubbles_mask[labeled_bubbles == region.label] = 255
        remaining_bubbles += 1
        remaining_area += area

# Compute the percentage of deleted bubbles and area erased
percentage_deleted = ((total_bubbles - remaining_bubbles) / total_bubbles) * 100
percentage_area_erased = ((total_area - remaining_area) / total_area) * 100

# Display the filtered mask
plt.figure(figsize=(8, 8))
plt.imshow(filtered_bubbles_mask, cmap='gray')
plt.title('Filtered Bubbles')
plt.axis('off')
plt.show()

# Apply the mask to the original image for visualization
final_filtered_result = np.where(filtered_bubbles_mask[..., None] == 255, 
                                 np.array([255, 255, 255], dtype=np.uint8), 
                                 img)

plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(final_filtered_result, cv2.COLOR_BGR2RGB))
plt.title('Final Filtered Bubbles')
plt.axis('off')
plt.show()

#print(f"Total bubbles: {total_bubbles}")
#print(f"Remaining bubbles: {remaining_bubbles}")
#print(f"Percentage of deleted bubbles: {percentage_deleted:.2f}%")
print(f"Total area before filtering: {total_area} pixels")
print(f"Remaining area after filtering: {remaining_area} pixels")
print(f"Percentage of area erased: {percentage_area_erased:.2f}%")


"-----------------------------------------------------------------------------------------------------"

labeled_bubbles = label(final_filled_mask)

props = regionprops(labeled_bubbles)

Area = []
Diameter = []
Real_Area = []

pixel_p = 0.001453171794174 # cm2/px2

for region in props:
    area = region.area
    Area.append(area)
    real_area = area * (pixel_p)
    Real_Area.append(real_area)
    diameter = 2 * np.sqrt(real_area/np.pi)
    Diameter.append(diameter)

"""
plt.figure(figsize=(8, 8))
plt.imshow(labeled_bubbles, cmap='nipy_spectral')
plt.title('Labeled Bubbles')
plt.axis('off')
plt.show()
"""




