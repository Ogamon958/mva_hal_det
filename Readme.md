Hallucination Detection using Multi-View Attention Features




truthfulQA
https://github.com/sylinrl/TruthfulQA


ragtruth
https://github.com/ParticleMedia/RAGTruth



CUDA Version
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.57.02    Driver Version: 470.57.02    CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA RTX A6000    On   | 00000000:01:00.0 Off |                  Off |
| 37%   65C    P2   181W / 300W |  37082MiB / 48685MiB |     90%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+


How to use

1.
1_create_data.ipynb


2.
2_ragtruth_preprocessing.ipynb


2_truthqa_preprocessing.ipynb



3.
3_training.ipynb
実行のたびにoptunaのハイパラ探索が異なるため，モデルの性能が少し異なる．



fine-tuned model


ragtruth

llama3-8b-instruct
https://huggingface.co/Ogamon/ragtruth_llama3_8b_instruct

qwen
https://huggingface.co/Ogamon/qwen-ragtruth


truthfulqa

llama3-8b-instruct
https://huggingface.co/Ogamon/llama-truthqa




Docker
Run Docker file in /docker
docker build -t tensorrt_jupyterlab .


docker run --name tensorrt --gpus all -p 8888:8888 \
  -v /path/to/home/directory/:/home/code/ \
  --shm-size=64g \
  -it tensorrt_jupyterlab


if kengen bug

out of container

id -u
#1009
id -g
#こちらも1009

in container
chown -R 1009:1009 /home/code