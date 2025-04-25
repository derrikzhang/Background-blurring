import os
import cv2
import numpy as np

# --- MediaPipe imports for segmentation ---
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

segmentation_model='./selfie_segmenter_square.tflite'
read_root = "./dataset/train/"
save_root = "./dataset/BGBlurDataset"
blur_kernel = 155
erode_size = 31
dilate_size = 31
alpha_size = 61

# Configure MediaPipe segmenter
BaseOptions  = python.BaseOptions(model_asset_path=segmentation_model)
VisionRunningMode = mp.tasks.vision.RunningMode
options  = vision.ImageSegmenterOptions(
base_options=BaseOptions,
output_category_mask=True
)   
segmenter = vision.ImageSegmenter.create_from_options(options)

def mediapipe_blur(img, blur_size=155,erode_size=31,dilate_size=31,alpha_size=61):
    """Blur background using MediaPipe Selfie Segmentation."""
    image = mp.Image.create_from_file(img)

    # Retrieve the category masks for the image
    segmentation_result = segmenter.segment(image)
    category_mask = segmentation_result.category_mask

    # Convert the BGR image to RGB
    image_data = cv2.cvtColor(image.numpy_view(), cv2.COLOR_BGR2RGB)
    mak = category_mask.numpy_view()
    
    #erode
    kernel = np.ones((erode_size,erode_size), dtype=np.int8)
    erode = cv2.erode(category_mask.numpy_view(), kernel, 3)
    kernel = np.ones((dilate_size,dilate_size), dtype=np.int8)
    dilate = cv2.dilate(erode, kernel, 1)

    # Apply effects
    blurred = cv2.GaussianBlur(image_data, (blur_size,blur_size), 0)

    diated_float = dilate.astype(np.float32) / 255.0
    alpha = cv2.dilate(diated_float,np.ones((11,11),np.uint8),iterations=1)
    alpha = cv2.GaussianBlur(alpha, (alpha_size,alpha_size), 0)
    alpha = np.clip(alpha, 0.0, 1)
    alpha_3c = np.stack([alpha] * 1, axis=-1)
    output_rgb = (blurred *alpha_3c + image_data * (1- alpha_3c)).astype(np.uint8)
    return output_rgb


if __name__ == "__main__":
    for root, dirs, files in os.walk(read_root):
        rel_path = os.path.relpath(root, read_root)
        dst_dir = os.path.join(save_root, rel_path)
        os.makedirs(dst_dir, exist_ok=True)
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                src_file = os.path.join(root, file)

                output = mediapipe_blur(src_file,
                                        blur_size=blur_kernel,
                                        erode_size=erode_size,
                                        dilate_size=dilate_size,
                                        alpha_size=alpha_size)
                
                relative_path = os.path.relpath(src_file, read_root)
                save_path = os.path.join(save_root, relative_path)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # dst_file = os.path.join(dst_dir, file)
                cv2.imwrite(save_path, output)
