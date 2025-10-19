
from fastapi import FastAPI
import uvicorn
from fastapi import APIRouter, UploadFile, File, HTTPException
import io
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image

classes = ['Cottage _Cabin', 'Mansion Luxury Villa', 'apartment_flat', 'hotel', 'single-family house']





class CheckImageRGB(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, 5),
        )

    def forward(self, x):
        x = self.first(x)
        x = self.second(x)
        return x

router = APIRouter(prefix="/rgb", tags=["RGB Model"])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_rgb = CheckImageRGB()
model_rgb.load_state_dict(torch.load("models/model_rgb (1).pth", map_location=device))
model_rgb.to(device)
model_rgb.eval()




transform_rgb = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])













@router.post("/predict/")
async def check_image_rgb(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="Файл кошулган жок.")

        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_tensor = transform_rgb(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model_rgb(img_tensor)
            pred = y_pred.argmax(dim=1).item()


        return {
            "class_id": pred,
            "class_name": classes[pred],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))