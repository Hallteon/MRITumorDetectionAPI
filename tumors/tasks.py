import math
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from celery import shared_task


@shared_task
def detect_brain_tumor(image_path):
    model = YOLO('tumor_detector.pt')
    model_classes = model.model.names

    with Image.open(image_path) as img:
        results = model(np.array(img))

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = box.cls[0]
                name = model_classes[int(cls)]

                draw = ImageDraw.Draw(img)
                draw.rectangle([x1, y1, x2, y2], outline="red")
                draw.text((max(0, x1), max(35, y1)), f'{name} 'f'{conf}')

        byteIO = BytesIO()
        img.save(byteIO, format='JPEG')
        byteIO.seek(0)
        encoded_image = base64.b64encode(byteIO.read()).decode('utf-8')

    os.remove(image_path)

    return encoded_image

