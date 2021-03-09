from __future__ import division
from __future__ import print_function

# self-defined functions
import model.crnn as crnn
from model import ocrutils 
import cv2
import torch
from torch.autograd import Variable
from PIL import Image
import string
import torchvision.transforms as transforms
import numpy as np
from scipy.special import softmax

#import pickle
#with open('model/weights/prior.pkl', 'rb') as f:
    #prior = pickle.load(f)
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(device)

def crnn_predict(crnn, img, transformer, decoder='bestPath', normalise=False):
    """
    Params
    ------
    crnn: torch.nn
        Neural network architecture
    transformer: torchvision.transform
        Image transformer
    decoder: string, 'bestPath' or 'beamSearch'
        CTC decoder method.
    
    Returns
    ------
    out: a list of tuples (predicted alphanumeric sequence, confidence level)
    """
    
    classes = string.ascii_uppercase + string.digits
    image = img.copy()
    
    image = transformer(image).to(device)
    image = image.view(1, *image.size())
    
    # forward pass (convert to numpy array)
    preds_np = crnn(image).data.cpu().numpy().squeeze()
    
    # move first column to last (so that we can use CTCDecoder as it is)
    preds_np = np.hstack([preds_np[:, 1:], preds_np[:, [0]]])
    
    preds_sm = softmax(preds_np, axis=1)
#     preds_sm = np.divide(preds_sm, prior)
    
    # normalise is only suitable for best path
    #if normalise == True:
        #preds_sm = np.divide(preds_sm, prior)
            
    if decoder == 'bestPath':
        output = ocrutils.ctcBestPath(preds_sm, classes)
        
    elif decoder == 'beamSearch':
        output = ocrutils.ctcBeamSearch(preds_sm, classes, None)
    else:
        raise Exception("Invalid decoder method. \
                        Choose either 'bestPath' or 'beamSearch'")
        
    return output

class AutoLPR:
    
    def __init__(self, decoder='bestPath', normalise=False):
        
        # crnn parameters
        self.IMGH = 32
        self.nc = 1 
        alphabet = string.ascii_uppercase + string.digits
        self.nclass = len(alphabet) + 1
        self.transformer = transforms.Compose([
            transforms.Grayscale(),  
            transforms.Resize(self.IMGH),
            transforms.ToTensor()])
        self.decoder = decoder
        self.normalise = normalise
        
                
    def load(self, crnn_path):

        # load CRNN
        self.crnn = crnn.CRNN(self.IMGH, self.nc, self.nclass, nh=256).to(device)
        self.crnn.load_state_dict(torch.load(crnn_path, map_location=device))
            
        # remember to set to test mode (otherwise some layers might behave differently)
        self.crnn.eval()
        
    def predict(self, img):
        # Convert cv2 format image to PIL format
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        self.image = im_pil
        return crnn_predict(self.crnn, self.image, self.transformer, self.decoder, self.normalise)
    
