from fastapi import APIRouter
from torch.nn.functional import poisson_nll_loss
from models.face import faceImage, faceImageEmbeddings
from utils.face.functions import detectWithMTCNN, face2embeddings, face_aligner, base64Img_cv2Img, array_to_base64, base64_to_array
import time

rounter = APIRouter()

@rounter.get('/')
def helloWorld():
    return 'API for face embeddings'

# For registration
# Input -> Face image (base64)
# Output -> base64 embeddings
@rounter.post('/mtcnnEmbeddings')
async def faceEmbeddings(inputs: faceImage):
    start_time = time.time()
    #MTCNN
    aligned_face = detectWithMTCNN(base64Img_cv2Img(inputs.image))
    #MobileFaceNet
    embeddings = face2embeddings(aligned_face)
    embeddings = embeddings.cpu().numpy()

    execution_time = time.time() - start_time
    result = {'Embedding': array_to_base64(embeddings), "Execution time": execution_time}
    return result

# For detections
# Input -> Cropped face image, box and points (all in base64 format)
# Output -> Embeddings
@rounter.post('/embeddings')
async def embeddings(inputs: faceImageEmbeddings):
    start_time = time.time()
    # Convert base64 back to array
    boxes = base64_to_array(inputs.boxes)
    points = base64_to_array(inputs.points)
    #Align face
    aligned_face = face_aligner(base64Img_cv2Img(inputs.image), boxes, points, image_size='112,112')
    #MobileFaceNet
    embeddings = face2embeddings(aligned_face)
    embeddings = embeddings.cpu().numpy()
    execution_time = time.time() - start_time
    result = {'Embedding': array_to_base64(embeddings), "Execution time": execution_time}
    return result
