import io
import base64
import numpy as np
import cv2
import base64
from utils.ocr.ppocr.predict_system import TextSystem

def base64Img_cv2Img(base64Img):
    # decode base64
    bytes_decoded = base64.b64decode(base64Img)
    # decode io
    buffer = io.BytesIO(bytes_decoded)
    # decode back to cv2
    decoded_img = cv2.imdecode(np.frombuffer(buffer.getbuffer(), np.uint8), -1)
    return decoded_img

def IntInString(input):
    try:
        int(input)
        return True
    except ValueError:
        return False

def recoginzed_plate(img):
    # Build PPOCR System
    sys = TextSystem()
    final_plate = []
    number = None
    results =  {'Plate Number': '', 'Confidence': '', 'Execution time': '', 'Plate Available': ''}
    # recognize plate
    dt_boxes, rec_res = sys(img)

    # Check if there is returned result
    if rec_res:
        for plate, conf in rec_res:
            # Post process to make sure number is on the back
            # Check if string is number
            if IntInString(plate):
                number = plate
            # append char on front
            else:
                final_plate.append(plate)
        # if there is number append on back of result
        if number:
            final_plate.append(number)
        # Join all list into string
        final_plate = ''.join(final_plate)
        # Remove space in car plate
        final_plate = final_plate.replace(" ", "")

        results['Plate Number'] = final_plate
        results['Confidence'] = round(float(conf), 2)
        results['Plate Available'] = True
        return results
    else:
        results['Plate Available'] = False
        return results