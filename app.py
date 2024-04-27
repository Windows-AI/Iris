apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6
from fastapi import FastAPI, UploadFile
import cv2
import numpy as np
from PIL import Image
import io
import base64
from celery import Celery

# Load YOLO
try:
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    net = None

layer_names = net.getLayerNames() if net else []
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()] if net else []

app = FastAPI()

celery_app = Celery('tasks', broker='pyamqp://guest@localhost//')

@celery_app.task
def process_image(data):
    image = Image.open(io.BytesIO(base64.b64decode(data)))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Process your image and predict the object here
    # description = model.predict(image)

    return description

@app.post('/image')
async def upload_image(file: UploadFile = File(...)):
    data = await file.read()
    result = process_image.delay(data)
    return {"task_id": str(result.id)}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
