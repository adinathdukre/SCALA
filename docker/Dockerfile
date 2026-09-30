FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ARG TASK
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/workspace \
    SCALA_TASK=${TASK} \
    SCALA_MODELS=/models \
    nnUNet_raw=/tmp/nnunet_raw \
    nnUNet_preprocessed=/tmp/nnunet_pre \
    nnUNet_results=/models \
    nnUNet_extTrainer=/workspace/scala/nnunet_ext

WORKDIR /workspace
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY models /models
COPY scala /workspace/scala
RUN python -m scala.predict --selftest

ENTRYPOINT ["python", "-m", "scala.predict"]
