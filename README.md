\# Deployment Architectures for Arabic Sentiment Analysis


Author
Souad Alharm

Asmarya Islamic University

Overview

This repository contains the code, deployment configurations, benchmarking scripts, and experimental artifacts used in the study of deployment architectures for transformer-based Arabic sentiment analysis services.



\## Models



The study evaluated the following pretrained Arabic language models:



\- AraBERT (aubmindlab/bert-base-arabertv02)

\- MARBERT (UBC-NLP/MARBERT)

\- CAMeLBERT-Mix (CAMeL-Lab/bert-base-arabic-camelbert-mix)



\## Dataset Summary



The dataset consisted of 423 manually labeled comments.



Label distribution:



\- Label 0: Positive sentiment (198 comments)

\- Label 1: Negative sentiment (225 comments)



The dataset is not distributed through this repository because it contains user-generated content and is subject to privacy and ethical considerations.



\## Training and Evaluation



The deployed checkpoint was trained using an 80/20 train-test split with:



\- random\_state = 42



Cross-validation experiments were performed using the following random seeds:



\- 11

\- 22

\- 33



Training and evaluation scripts are available in the `classification/` directory.



\## Deployment Artifacts



The repository contains:



\- FastAPI inference service

\- Dockerfile and dependency specifications

\- Docker Compose deployment configurations

\- Nginx reverse-proxy configurations

\- Locust load-testing scripts

\- Experiment automation scripts



Deployment configurations are provided in the `deployment/` directory.



\## Experiment Results



Archived experiment outputs are available in:



\- results\_main.zip

\- results\_tuned.zip

\- results\_w200.zip



These archives contain benchmark outputs and analysis artifacts used in the study.



Large raw Nginx logs are intentionally excluded.



\## Model Checkpoint Fingerprint



SHA-256:



FCCD06909102A7B3C095F813996EF64B3F4634860AECB9C452E3BD64776EAA35



\## Environment



Serving environment:



\- Python 3.10

\- FastAPI

\- PyTorch 2.2.2 (CPU-only)

\- Docker

\- Nginx

\- Locust



\## Reproducing the Experiments



Start a deployment configuration:



```bash

docker compose up -d

```



Run the load tests:



```bash

locust -f loadtest/locustfile.py

```



Run the analysis scripts:



```bash

python classification/analyze\_cv.py

```



\## Hardware Notes



Performance measurements were collected on the hardware platform described in the accompanying paper.



Absolute throughput and latency values may vary across machines, operating systems, Docker versions, and execution sessions. The reported values should therefore be interpreted primarily for relative comparison between deployment architectures rather than as universal performance numbers.



\## Data Availability



The dataset used in this study is not publicly distributed because it contains user-generated comments and is subject to privacy and ethical considerations.



\## License



Source code is released under the license provided in this repository.

