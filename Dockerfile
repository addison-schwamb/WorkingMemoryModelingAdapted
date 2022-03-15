# syntax=docker/dockerfile:1
FROM python:3.9-slim

RUN pip install numpy
RUN pip install scipy
RUN pip install matplotlib
RUN pip install drawnow

COPY allDigCNNMNIST .
COPY posthoc_tests.py .
COPY robustness_tests.py .
COPY SPM_task.py .
COPY train_force.py .
COPY train_posthoc_clst.py .
COPY train_input.py .
COPY damage_network.py .