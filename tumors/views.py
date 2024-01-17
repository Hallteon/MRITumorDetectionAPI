import cv2
import uuid
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView
from tempfile import NamedTemporaryFile
from tumors.tasks import detect_brain_tumor


class TumorDetectView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    def post(self, request, format=None):
        image_data = request.FILES['image']
        temp_file = NamedTemporaryFile(delete=False)

        for chunk in image_data.chunks():
            temp_file.write(chunk)

        processed_image = detect_brain_tumor.delay(temp_file.name)

        return Response(processed_image.get(), status=status.HTTP_201_CREATED, content_type="text/plain")
