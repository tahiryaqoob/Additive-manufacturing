import os
import tifffile
import imageio
import numpy as np
from PIL import Image
import cv2

folder_path = r"C:\Users\tahir\Desktop\Python\Tiff_data"
output_folder = r"C:\Users\tahir\Desktop\Python\Processed_data"

upper_box = [148, 58, 294, 158]
middle_box = [140, 520, 294, 620]
lower_box = [140, 980, 294, 1100]

files = os.listdir(folder_path)

def process_picture(image, left, upper, right, lower):
	# Crop the image using OpenCV
	cropped_image = image[upper:lower, left:right]
	preimage = cv2.GaussianBlur(cropped_image, (7, 7), 0)

	# Remove any small white noise pixels
	preimage = cv2.erode(preimage, np.ones((1,1)), iterations=1)

	# Set pixel values greater than 150 to 255
	#reimage[preimage > 20] = 225

   # Rotate the cropped image using OpenCV
	rotated_image = cv2.rotate(preimage, cv2.ROTATE_90_CLOCKWISE)

	return rotated_image

def calculate_pixel_statistics(folder_path):
	files = os.listdir(folder_path)
	pixel_mean_intensity = []
	pixel_std_dev_intensity = []

	for file in files:
		image_path = os.path.join(folder_path, file)
		with Image.open(image_path) as image:
			image_array = np.array(image)
			image_intensity = image_array.astype(np.uint8) 

		pixel_mean_intensity.append(image_intensity)
		pixel_std_dev_intensity.append(np.std(image_intensity)) 

	return pixel_mean_intensity, pixel_std_dev_intensity

for i in files:
	folder_name = i.replace('.tif', '')
	image_path = os.path.join(folder_path, i)
	
	# Create output directories if they don't exist
	for subfolder in ["Cropped_Tif", "Full_Pictures", "Statistics"]:
		os.makedirs(os.path.join(output_folder, subfolder, folder_name), exist_ok=True)
		for sub_subfolder in ["First", "Second", "Third"]:
			if subfolder == "Cropped_Tif":
				os.makedirs(os.path.join(output_folder, subfolder, folder_name, sub_subfolder), exist_ok=True)
	
	with tifffile.TiffFile(image_path) as tif:
		image = tif.series[0].asarray()
		height, width, num_slices = image.shape

	max_intensity = np.max(image)
	min_intensity = np.min(image)

	# Scale image
	scaled_image = ((image - min_intensity) / (max_intensity - min_intensity)) * 65535

	for n in range(num_slices):
		output_file_path = os.path.join(output_folder, "Full_Pictures", folder_name, f'{folder_name}_slice_{n+1}.png')
		image_slice = scaled_image[:, :, n].astype(np.uint16)
		imageio.imwrite(output_file_path, image_slice)

		# Process the image using OpenCV
		image_cv = cv2.imread(output_file_path, cv2.IMREAD_UNCHANGED)
		rotated_image1 = process_picture(image_cv, *upper_box)
		rotated_image2 = process_picture(image_cv, *middle_box)
		rotated_image3 = process_picture(image_cv, *lower_box)

		# Save processed images using OpenCV
		cv2.imwrite(os.path.join(output_folder, "Cropped_Tif", folder_name, "First", f'{folder_name}_first_{n+1}.png'), rotated_image1)
		cv2.imwrite(os.path.join(output_folder, "Cropped_Tif", folder_name, "Second", f'{folder_name}_second_{n+1}.png'), rotated_image2)
		cv2.imwrite(os.path.join(output_folder, "Cropped_Tif", folder_name, "Third", f'{folder_name}_third_{n+1}.png'), rotated_image3)

	# Calculate statistics
	first_mean_intensity, first_std_dev_intensity = calculate_pixel_statistics(os.path.join(output_folder, "Cropped_Tif", folder_name, "First"))
	second_mean_intensity, second_std_dev_intensity = calculate_pixel_statistics(os.path.join(output_folder, "Cropped_Tif", folder_name, "Second"))
	third_mean_intensity, third_std_dev_intensity = calculate_pixel_statistics(os.path.join(output_folder, "Cropped_Tif", folder_name, "Third"))

	# Write statistics to text files
	with open(os.path.join(output_folder, "Statistics", folder_name, f'{folder_name}_mean_intensity_first.txt'), 'w') as f:
		f.write('\n'.join(map(str, first_mean_intensity)) + '\n')
	with open(os.path.join(output_folder, "Statistics", folder_name, f'{folder_name}_mean_intensity_second.txt'), 'w') as f:
		f.write('\n'.join(map(str, second_mean_intensity)) + '\n')
	with open(os.path.join(output_folder, "Statistics", folder_name, f'{folder_name}_mean_intensity_third.txt'), 'w') as f:
		f.write('\n'.join(map(str, third_mean_intensity)) + '\n')

	with open(os.path.join(output_folder, "Statistics", folder_name, f'{folder_name}_std_dev_intensity_first.txt'), 'w') as f:
		f.write('\n'.join(map(str, first_std_dev_intensity)) + '\n')
	with open(os.path.join(output_folder, "Statistics", folder_name, f'{folder_name}_std_dev_intensity_second.txt'), 'w') as f:
		f.write('\n'.join(map(str, second_std_dev_intensity)) + '\n')
	with open(os.path.join(output_folder, "Statistics", folder_name, f'{folder_name}_std_dev_intensity_third.txt'), 'w') as f:
		f.write('\n'.join(map(str, third_std_dev_intensity)) + '\n')

