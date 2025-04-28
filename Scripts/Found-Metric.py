import cv2
import numpy as np
import matplotlib.pyplot as plt
import statistics

# Load the image
img = cv2.imread(r'C:\Users\juanp\Descargas\p_12.jpg')

# Split the image into its color channels
img_B, img_G, img_R = cv2.split(img)

# Apply adaptive thresholding to the blue channel
binImg = cv2.adaptiveThreshold(img_B, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Apply morphological closing to close gaps in the grid lines
kernel = np.ones((5, 5), np.uint8)
closed_binImg = cv2.morphologyEx(binImg, cv2.MORPH_CLOSE, kernel)

# Detect contours
contours, _ = cv2.findContours(closed_binImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image to draw the contours on
image_with_contours = cv2.cvtColor(closed_binImg, cv2.COLOR_GRAY2BGR)  # Convert to BGR for color display

# Draw the contours on the image
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)  # Draw contours in green with thickness 2

# Display the image with the drawn contours
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(image_with_contours, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
plt.title('Borders Detected')
plt.axis('off')
plt.show()

# Filter contours to detect squares and find their areas
filtered_contours = []
areas_pixel = []  # List to store areas of the squares

for cnt in contours:
    # Approximate contour to polygon
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    # Check if polygon has 4 vertices
    if len(approx) == 4:
        # Compute bounding rectangle and aspect ratio
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        area = cv2.contourArea(cnt)
        
        # Filter based on aspect ratio and area
        if 0.9 <= aspect_ratio <= 1.1 and area > 50:
            filtered_contours.append(approx)
            areas_pixel.append(area)  # Append area to the list

real_area = 1.0  # cm² for each square
print(f"Real Area: {real_area:.2f} cm2")
average_area_pixels = sum(areas_pixel) / len(areas_pixel)
print(f"Average area in pixels: {average_area_pixels:.2f} pixels 2")

escaling_factor = real_area / average_area_pixels
print(f"Escaling factor: {escaling_factor:.15f} cm2/px2")

print(statistics.stdev(areas_pixel))

# Draw and fill the filtered contours
img_borders_and_fill = img.copy()
cv2.drawContours(img_borders_and_fill, filtered_contours, -1, (0, 255, 0), thickness=2)
cv2.drawContours(img_borders_and_fill, filtered_contours, -1, (0, 255, 0), thickness=cv2.FILLED)

# Display the result
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(img_borders_and_fill, cv2.COLOR_BGR2RGB))
plt.title('Borders and Filled Grid Contours')
plt.axis('off')
plt.show()
