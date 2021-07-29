# Install torch-cpu
pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Install torch-gpu
# pip3 install torch torchvision

# Install third party libs
pip3 install -r requirements.txt

# Download weights
# PPOCR
gdown https://drive.google.com/uc?id=1JKrXpZz0jEKMnFtQOmGVCidnaHBOxXjm -O utils/ocr/ppocr/weights/ch_ppocr_mobile_v2.0_rec_infer/inference.pdmodel
gdown https://drive.google.com/uc?id=1MMZzkmMBGihzhGCUYEeatDfDmNbjDtge -O utils/ocr/ppocr/weights/ch_ppocr_mobile_v2.0_rec_infer/inference.pdiparams.info
gdown https://drive.google.com/uc?id=1HRcqA1WGoSw0O8T-B3Yc1NJfbkrdfKOq -O utils/ocr/ppocr/weights/ch_ppocr_mobile_v2.0_rec_infer/inference.pdiparams

gdown https://drive.google.com/uc?id=12kEs81PJP0A8kTgDkzF4CgstKpR62f9A -O utils/ocr/ppocr/weights/ch_ppocr_mobile_v2.0_det_infer/inference.pdmodel
gdown https://drive.google.com/uc?id=1g23GgB-aTI4eUDqWFNZJ1IKlqwCC5pcN -O utils/ocr/ppocr/weights/ch_ppocr_mobile_v2.0_det_infer/inference.pdiparams.info
gdown https://drive.google.com/uc?id=1FYQevkKX0_NEqJfYIuhOa2wtWw2Imss5 -O utils/ocr/ppocr/weights/ch_ppocr_mobile_v2.0_det_infer/inference.pdiparams


# Face 
gdown https://drive.google.com/uc?id=15361IIJU3tPRJ2KD_X6tKaqwGwyh5C1_ -O utils/face/embeddings/weights/rnet.pt
gdown https://drive.google.com/uc?id=1Ca4xxZ3_iq2vEScEwJtOdCG9BJczapTq -O utils/face/embeddings/weights/pnet.pt
gdown https://drive.google.com/uc?id=15ZyCbvUCDWVQNFlBKet1PbTb_3zbUexX -O utils/face/embeddings/weights/onet.pt
gdown https://drive.google.com/uc?id=1v0dyHpMHFxuMnvGsKOHRhRzuGOj0u76u -O utils/face/embeddings/weights/mobilefacenet.pth