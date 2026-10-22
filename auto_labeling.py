import os
import glob
from ultralytics import YOLO

# 1. Load the pre-trained standard YOLOv8 model
print("Loading pre-trained YOLOv8m model...")
model = YOLO('yolov8m.pt') 

# 2. Point this to your folder of 5,249 raw images
IMAGE_DIR = "house_data"
images = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))

if not images:
    print("No images found! Check your folder path.")
    exit()

# 3. Map YOLO's built-in COCO IDs to YOUR custom IDs
# YOLO already knows these objects. We map them to 0, 1, 2, 3, 4 so you can train your specific home automation model.
class_mapping = {
    62: 0, # tv
    63: 1, # laptop
    67: 2, # cell phone
    68: 3, # microwave
    60: 4  # dining table
}

print(f"Found {len(images)} images. Blasting through them with the GPU...")

# 4. Loop through every image and let the AI generate the .txt files
for img_path in images:
    # Run the image through the pre-trained model silently using the GPU
    results = model(img_path, verbose=False, device=0)[0]
    
    txt_filename = os.path.splitext(img_path)[0] + ".txt"
    
    with open(txt_filename, "w") as f:
        # Loop through every object the model found in this specific image
        for box in results.boxes:
            coco_class_id = int(box.cls[0])
            
            # If the object is one of your target classes, save it!
            if coco_class_id in class_mapping:
                custom_class_id = class_mapping[coco_class_id]
                
                # Extract the normalized YOLO coordinates (center_x, center_y, width, height)
                coords = box.xywhn[0].tolist() 
                
                # Write the bounding box to the text file
                f.write(f"{custom_class_id} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}\n")

print("\n🎉 Auto-Labeling Complete! All .txt files have been generated.")