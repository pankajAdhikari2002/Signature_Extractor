import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('images/test_img2.jpg')

# Output Path for the resultant image
output_path = "output/output_img.png"

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
_, black_white_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Inverse
black_white_image = cv2.bitwise_not(black_white_image)

# Find contours
contours, _ = cv2.findContours(black_white_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank canvas with an alpha channel
canvas = np.zeros_like(image, dtype=np.uint8)
canvas.fill(255)  # Fill with fully opaque white

# Draw contours on the canvas with transparent background
cv2.drawContours(canvas, contours, -1, (0, 255, 0), thickness=cv2.FILLED)  # Fill contours with green color

# Convert canvas to RGBA (add alpha channel)
canvas_rgba = cv2.cvtColor(canvas, cv2.COLOR_BGR2BGRA)

# Set alpha channel to 0 inside the contour
for contour in contours:
    cv2.fillPoly(canvas_rgba, [contour], (0, 0, 0, 0))  # Set alpha channel to 0 inside the contour

# Morphological operations to thin out the contour lines
kernel = np.ones((10, 10), np.uint8)
canvas_rgba = cv2.morphologyEx(canvas_rgba, cv2.MORPH_OPEN, kernel)
canvas_rgba = cv2.morphologyEx(canvas_rgba, cv2.MORPH_CLOSE, kernel)


# Save the canvas with transparent background
success = cv2.imwrite(output_path, canvas_rgba)

if success:
    print("Output saved as:", output_path)
else:
    print("Failed to save the output.")

# View
# plt.imshow(canvas_rgba)
# plt.axis('off')
# plt.show()
