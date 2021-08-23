from fastapi import APIRouter
from models.face import faceImage, faceImageEmbeddings, comparisons
from utils.face.functions import detectWithMTCNN, face2embeddings, face_aligner, base64Img_cv2Img, array_to_base64, base64_to_array, compare_faces
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

# For comparing embeddings
# Input -> Target embeddings and List of existing embeddings
# output -> matched and userId
@rounter.post('/compare')
async def comparing(inputs: comparisons):
    # preprocess data
    listofEmbs = []
    listsofUsers = []
    listsOfdata = inputs.embeddings
    for emb in listsOfdata:
        listofEmbs.append(base64_to_array(emb["embeddings"])[0])
        listsofUsers.append(emb["userId"])
    targetEmb = base64_to_array(inputs.embedding_to_compare)
    # comparing
    matched, userId, min_dist = compare_faces(lists=listofEmbs, target=targetEmb, names=listsofUsers)
    # result
    result = {"matched": matched, "userId": userId, "min_dist": str(min_dist)}
    return result
