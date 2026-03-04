from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import List
from model import FusionModel2
from typing import Annotated

import torch
from PIL import Image
import numpy as np
import io
import json
from torch import nn
from torchvision import transforms

app = FastAPI()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

model = FusionModel2().to(DEVICE)
model.load_state_dict(torch.load("checkpoints/model2_epoch_20.pth", map_location=DEVICE))
model.eval()

@app.get("/")
def home():
    return {"status" : "Running", "project" : "Vision API"}

@app.post('/perdict_sequence')
async def perdict_sequence(files: Annotated[list[UploadFile], File(...)],
                           boxes_raw : Annotated[str, Form(...)]):
    if len(files) != 10:
        raise HTTPException(status_code = 400, detail = "Not enough files uploaded")
    
    # for i, file in enumerate(files):
    #     data = await file.read()
    #     with open(f"000{i}.png", "wb") as f:
    #         file.write(data)
    
    boxes = json.loads(boxes_raw)
    input_tensor = []
    for i in range(10):
        file = await files[i].read()
        l, t, r, b = boxes[i]
        I = Image.open(io.BytesIO(file)).convert("RGB")
        add_w = (r - l) * 0.3
        add_h = (b - t) * 0.3

        crop_l = max(0, l - add_w)
        crop_t = max(0, t - add_h)
        crop_r = min(r + add_w, I.width)
        crop_b = min(b + add_h, I.height)

        crop_box = (crop_l, crop_t, crop_r, crop_b)

        crop_I = I.crop(crop_box)
        input_tensor.append(transform(crop_I))
    
    input_batch = torch.stack(input_tensor).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        predict = model(input_batch)
    
    return {"status" : "success",
            "predict" : predict.cpu().tolist()}