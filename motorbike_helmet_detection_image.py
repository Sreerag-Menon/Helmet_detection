import os
import cv2
import numpy as np
from bike_helmet_detector_image import detection
from utils import visualization_utils as vis_util

# Define the folder path containing the images
folder_path = '/home/sreerag/project/Motorbike-Helmet_detection-using-YOLOV3/images'

# Define the folder to save the results
results_folder = 'results'

# Create the results folder if it doesn't exist
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

# Get a list of all image files in the folder
image_files = [file for file in os.listdir(folder_path) if file.endswith(('.jpg', '.jpeg', '.png'))]
print(folder_path)
# Loop through each image file
for image_file in image_files:
    # Get the full path of the image
    image_path = os.path.join(folder_path, image_file)

    # Get the category index, images, boxes, scores and classes for helmet
    category_index_helmet, image_helmet, boxes_helmet, scores_helmet, classes_helmet, num_helmet = \
        detection('frozen_graphs', '/frozen_inference_graph_helmet.pb', '/labelmap_helmet.pbtxt', 2, image_path)

    # Get the category index, images, boxes, scores and classes for motorbike and person
    category_index_motorbike, image_motorbike, boxes_motorbike, scores_motorbike, classes_motorbike, num_motorbike = \
        detection('frozen_graphs', '/frozen_inference_graph_motorbike.pb', '/labelmap_motorbike.pbtxt', 4, image_path)

    # Visualize helmet detection
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_helmet,
        np.squeeze(boxes_helmet),
        np.squeeze(classes_helmet).astype(np.int32),
        np.squeeze(scores_helmet),
        category_index_helmet,
        use_normalized_coordinates=True,
        line_thickness=6,
        min_score_thresh=0.70
    )

    # Visualize motorbike detection
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_motorbike,
        np.squeeze(boxes_motorbike),
        np.squeeze(classes_motorbike).astype(np.int32),
        np.squeeze(scores_motorbike),
        category_index_motorbike,
        use_normalized_coordinates=True,
        line_thickness=6,
        min_score_thresh=0.75
    )

    # Save the images with detection results into the results folder
    cv2.imshow('Object detector', image_helmet)
    cv2.imwrite(os.path.join(results_folder, f'{image_file}_detection.jpg'), image_helmet)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
