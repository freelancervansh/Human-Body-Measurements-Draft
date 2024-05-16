from PIL import Image
from fastapi.params import Depends
from fastapi import FastAPI, HTTPException, Request, Response

import torch
from predictor import *
import logging
import traceback


app = FastAPI()
Estimation_model_path = "../models/chest_estimation_model.pth"


model = ChestWidthPredictor()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.load_state_dict(torch.load(Estimation_model_path,map_location ='cpu'))

async def get_body(request: Request):
    return await request.json()


@app.get("/")
async def check():
    return {
        "status" : "alive"
    }

@app.get("/v1/health")
async def check():
    return {
        "status" : "alive"
    }

@app.post("/v1/models/DL_model/predict")
async def predictions(body: dict=Depends(get_body)):

    try : 

        '''
        request -- > {
        "unique_id": str,
        "front_image_url": url,
        "side_image_url": url,
        "gender": str,
        "height": str (in cm)
        }
        '''
        unique_id = body.get("unique_id")
        front_image_url = body.get("front_image")
        side_image_url = body.get("side_image")
        gender = gender_dict.get(body.get("gender"))
        height = float(body.get("height"))

        result = {
                        "predictions" : {"chest":None},
                        "is_success": 1,
                        "error": ""
                    }

        front_image = load_image_from_url(front_image_url)
        # front_image = get_image_from_path(front_image_url)
    
        if front_image is None:
            result['is_success'] = 0
            result['error'] = "Cannot fetch front image from url"
            return result
        
        side_image = load_image_from_url(side_image_url)
        # side_image = get_image_from_path(side_image_url)
        if side_image is None:
            result['is_success'] = 0
            result['error'] = "Cannot fetch side image from url"
            return result

        front_image = np.asarray(front_image)
        side_image = np.asarray(side_image)
        front_image = segmentation(front_image)
        side_image = segmentation(side_image)

        front_image = transform(Image.fromarray(front_image)).unsqueeze(0).to(device).float()
        side_image = transform(Image.fromarray(side_image)).unsqueeze(0).to(device).float()

        gender = torch.tensor(gender, dtype=torch.float).unsqueeze(0).to(device)  
        height = torch.tensor(height, dtype=torch.float).unsqueeze(0).to(device)

        outputs = model(front_image, side_image, gender, height) # to numpy
        
        result['predictions']["chest"] = outputs.item()//2.54

        return result

    except Exception as e:
        result = {
                        "predictions" : {"chest":None},
                        "is_success": 0,
                        "error": str(traceback.format_exc())
                    }
        return result


@app.post("/v1/models/estimation_model/predict")
async def predictions(body: dict=Depends(get_body)):

    try : 
        unique_id = body.get("unique_id")
        front_image_url = body.get("front_image")
        side_image_url = body.get("side_image")
        gender = gender_dict.get(body.get("gender"))
        height = float(body.get("height"))

        result = {
                        "predictions" : {"chest":None},
                        "is_success": 1,
                        "error": ""
                    }

        front_image = load_image_from_url(front_image_url)
        # front_image = get_image_from_path(front_image_url)
    
        if front_image is None:
            result['is_success'] = 0
            result['error'] = "Cannot fetch front image from url"
            return result
        
        side_image = load_image_from_url(side_image_url)
        # side_image = get_image_from_path(side_image_url)
        if side_image is None:
            result['is_success'] = 0
            result['error'] = "Cannot fetch side image from url"
            return result

        front_image = np.asarray(front_image)
        side_image = np.asarray(side_image)
        front_image_seg = segmentation(front_image)
        side_image_seg = segmentation(side_image)
        front_image_seg = (front_image_seg > 0).astype(int)
        side_image_seg = (side_image_seg > 0).astype(int)

        chest_width_pixels, _, _ = estimate_chest_width(front_image,'front')
        chest_keypoint = estimate_chest_width(side_image,'side')
        chest_depth_pixels = estimate_chest_measurements(side_image_seg,chest_keypoint)
        ## find height of person
        person_height_pixels = calculate_height_from_segmentation(front_image_seg)

        # Convert chest width from pixels to centimeters
        chest_width_cm = pixels_to_cm(chest_width_pixels,person_height_pixels,height)
        chest_depth_cm = pixels_to_cm(chest_depth_pixels,person_height_pixels,height)

        chest_circumference_cm = 2 * np.pi * ((chest_width_cm + chest_depth_cm) / 4)

        
        result['predictions']["chest"] = chest_circumference_cm//2.54

        return result

    except Exception as e:
        result = {
                        "predictions" : {"chest":None},
                        "is_success": 0,
                        "error": str(traceback.format_exc())
                    }
        return result


#
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("app:main", host="0.0.0.0", port=8000, reload=True)
    