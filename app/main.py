from model_test import predict_image
from config import CONFIDENT_SCORE, NUM_CLASSES, MODEL_PATH
from model import get_model_instance_segmentation
from PIL import Image
from fastapi import FastAPI, UploadFile
import torch
import torchvision.transforms.functional as F
 
# Creating FastAPI instance
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Model is online"}
 
@app.post("/predict/") 
async def predict_upload_file(file: UploadFile):
    # Get model
    model = get_model_instance_segmentation(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    
    # Open image
    image = Image.open(file.file)
    image = F.pil_to_tensor(image)                  # convert to tensor shape (3,640,640)
    image_tensor = F.convert_image_dtype(image)     # covert to type for the model
    
    # prediction
    pred_boxes, pred_labels = predict_image(model, image_tensor, CONFIDENT_SCORE)
    return {
        "boxes": pred_boxes,
        "labels": pred_labels
    }