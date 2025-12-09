import os

import numpy as np
import onnxruntime as ort
from io import BytesIO
from urllib import request
from PIL import Image

model_name = os.getenv("MODEL_NAME", "hair_classifier_empty.onnx")

target_size= (200, 200)


session = ort.InferenceSession(
    model_name, providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def to_tensor_no_torch(img):
    """Convert PIL image → CHW float32 numpy array in [0,1]."""
    arr = np.array(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return arr


def normalize_no_torch(arr, mean, std):
    """Apply channel-wise normalization on numpy array."""
    arr = (arr - np.array(mean)[:, None, None]) / np.array(std)[:, None, None]
    return arr

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)

    # Convert to CHW numpy
    img = to_tensor_no_torch(img)

    # Manual normalization (no PyTorch required)
    img = normalize_no_torch(
        img,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Add batch dimension → (1,3,H,W)
    img = np.expand_dims(img, axis=0)

    return img.astype("float32")

    return img

def predict(url):
    img=download_image(url)
    imgT=prepare_image(img,target_size)
    result = session.run([output_name], {input_name: imgT})
    python_output = result[0].tolist()

    return {"result": python_output}

def lambda_handler(event, context):
    url = event["url"]
    result = predict(url)
    return result