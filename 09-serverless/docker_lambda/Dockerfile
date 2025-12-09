FROM agrigorev/model-2025-hairstyle:v1

RUN pip install numpy pillow onnxruntime --no-cache-dir
RUN pip install torchvision --no-deps --no-cache-dir


COPY lambda_function.py ./

CMD ["lambda_function.lambda_handler"]