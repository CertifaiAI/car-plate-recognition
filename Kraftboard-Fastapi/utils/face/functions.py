import base64
import io
import cv2
import numpy as np
from utils.face.embeddings.models import MTCNN, MobileFaceNet
from PIL import Image
from torchvision import transforms
import torch
import pickle
from core.config import Settings

Settings = Settings()

mtcnn = MTCNN(keep_all=True, thresholds=[0.7,0.8,0.8], min_face_size=100, device=Settings.DEVICE)
mobile_facenet = MobileFaceNet(pretrained=True, training=False, device=Settings.DEVICE)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def base64Img_cv2Img(base64Img):
    # decode base64
    bytes_decoded = base64.b64decode(base64Img)
    # decode io
    buffer = io.BytesIO(bytes_decoded)
    # decode back to cv2
    decoded_img = cv2.imdecode(np.frombuffer(buffer.getbuffer(), np.uint8), -1)
    return decoded_img

def face_aligner(img, bbox=None, landmark=None, **kwargs):
    M = None
    image_size = []

    # image_size='112,112'
    str_image_size = kwargs.get('image_size', '')

    # if img size defined 
    if len(str_image_size) > 0:
        image_size = [int(x) for x in str_image_size.split(',')]
        if len(image_size) == 1:
            image_size = [image_size[0], image_size[0]]

        # Checking if bellow statement is true
        assert len(image_size) == 2
        assert image_size[0] == 112
        assert image_size[0] == 112 or image_size[1] == 96

    # if got landmark -> M
    if landmark is not None:
        assert len(image_size) == 2
        # Fixed point of face landmarks
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        if image_size[1] == 112:
            src[:, 0] += 8.0
        dst = landmark.astype(np.float32)

        # tform = trans.SimilarityTransform()
        # tform.estimate(dst, src)
        # M = tform.params[0:2, :]
        # M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)
        # M = cv2.estimateAffinePartial2D(dst, src, False)

        # Stretch the landmarks to fixed point of src
        M = cv2.estimateAffinePartial2D(dst, src, method=cv2.LMEDS)
        M = M[0]

    # if no landmark
    if M is None:
        # if bbox is none -> assume box is ald cropped
        if bbox is None:  # use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(image_size) > 0:
            # Resize image to 112,112
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    else:  # do align using landmark
        assert len(image_size) == 2

        # src = src[0:3,:]
        # dst = dst[0:3,:]

        # print(src.shape, dst.shape)
        # print(src)
        # print(dst)
        # print(M)
        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)

        # tform3 = trans.ProjectiveTransform()
        # tform3.estimate(src, dst)
        # warped = trans.warp(img, tform3, output_shape=_shape)
        return warped

def detectWithMTCNN(image):
    # Preprocess 
    face_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_img = cv2.resize(image, (600,900))
    boxes, _, points = mtcnn.detect(face_img, landmarks=True)
    print(points)
    print(boxes)
    aligned_face = face_aligner(face_img, boxes, points, image_size='112,112')
    return aligned_face

def face2embeddings(aligned_face):
    with torch.no_grad():
        image = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
        image = Image.fromarray(image)
        image = transform(image).to('cpu').unsqueeze(0)
        return mobile_facenet(image)

def array_to_base64(array):
    bytes_array = pickle.dumps(array)
    array_base64 = base64.b64encode(bytes_array).decode('ascii')
    return array_base64

def base64_to_array(base64_string):
    bytes_decoded = base64.b64decode(base64_string)
    return pickle.loads(bytes_decoded)

def list2array(list):
    return (np.array(list))
