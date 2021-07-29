# Install torch
bash install_torch.sh

# Install third party libraries
pip install -r requirements.txt

# Download weights
gdown https://drive.google.com/uc?id=18tyNWkGC_x9FddZ9hJ5di3_Sc9WPkpd_ -O yolov5/weights/detection.pt