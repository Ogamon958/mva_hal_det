# Hallucination Detection using Multi-View Attention Features

## Research Overview

This research focuses on detecting hallucinations in Large Language Models (LLMs) at the **token level**.

- A Transformer encoder-based model is trained via **supervised learning** to predict whether each token contains hallucination.

- **RAGTruth** is used as the labeled dataset for training and evaluation.

- Additionally, experiments are conducted under a **sentence-level hallucination detection** setting using **TruthfulQA**.

RAGTruth
https://github.com/ParticleMedia/RAGTruth


TruthfulQA
https://github.com/sylinrl/TruthfulQA



Fine-tuned Model

RAGTruth

llama3-8b-instruct
https://huggingface.co/Ogamon/ragtruth_llama3_8b_instruct

qwen
https://huggingface.co/Ogamon/qwen-ragtruth


TruthfulQA

llama3-8b-instruct
https://huggingface.co/Ogamon/llama-truthqa



- **Environment**
    - **CUDA Version**: 11.4  
        - **Driver Version**: 470.57.02  
        - *Note: The CUDA version is slightly outdated, so some modifications may be required for newer versions.*
    - **GPU**: NVIDIA RTX A6000, up to 3 GPUs (48GB each)
    - **Docker Version**: 20.10.8  
        - **API Version**: 1.41

        

- **Docker Environment Setup**
    - To build the image, run the following command in the directory where the Dockerfile is located (`/docker`):

        ```
        docker build -t tensorrt_jupyterlab .
        ```

        - If you want to update the libraries, please modify `/docker/requirements.txt` before running the above command.

    - Then, run the following command to start the container:

        ```
        docker run --name tensorrt --gpus all -p 8888:8888 \
          -v /path/to/home/directory/:/home/code/ \
          --shm-size=64g \
          -it tensorrt_jupyterlab
        ```

        - This code uses Jupyter Notebook.
        - Replace `/path/to/home/directory/` with the full path of the directory you want to mount (e.g., `/home/user/hallucination/Hal_detect_attention`).
        - The `--name tensorrt` part is the container name; change it as needed.
        - Use `--gpus` to specify which GPUs to use.
        - If port 8888 is already in use, please change it.
        - The `code/data` directory in this project may include large files (hundreds of GB).
            - If storing such files on the main drive may cause space issues, follow these steps:

                1. Copy the `data` directory to a large-capacity drive.
                2. Mount the copied directory using the command below:

                    ```
                    docker run --name tensorrt --gpus all -p 8888:8888 \
                      -v /path/to/home/directory/:/home/code/ \
                      -v /path/to/data/:/home/code/data/ \
                      --shm-size=64g \
                      -it tensorrt_jupyterlab
                    ```

                    - Set `/path/to/data/` to the full path of the copied data directory.

                    **Example (for user):**

                    ```
                    docker run --name tensorrt --gpus all -p 8889:8888 \
                      -v /home/user/hallucination/Hal_detect_attention/:/home/code/ \
                      -v /data1/user/hallucination/data:/home/code/data/ \
                      -v /home/user/vishnu:/home/code/vishnu \
                      --shm-size=64g \
                      -it tensorrt_jupyterlab
                    ```

                    - `/home/user/hallucination/Hal_detect_attention` is the main drive.
                    - `/data1/user/hallucination/data` is a drive that can store large data.
                    - The left side of the `:` is the local path.
                    - `vishnu` is a file server.

    - If you are using VS Code:
        - You can start the container from the **Remote Explorer** section

        - You can operate and run Jupyter Notebook inside the container using VS Code:

        - To run Python code in VS Code, make sure the required extensions are installed:
            - Install **Python** and **Jupyter** extensions.
            - Optionally, install other useful extensions like **Copilot**.
            - If it doesn't work properly, try the following steps in order:
                1. Close and reopen the VS Code window.
                2. Restart the container.
                3. Rebuild and restart the container.

    
    

## Description of Main Code (located in `/features`)

### `1_create_data.ipynb`
Creates attention-based features:

- **key_avg**: Average attention received by each token from all other tokens.  
- **key_entropy**: Diversity of tokens attending to each token.  
- **query_entropy**: Diversity of tokens each token attends to during generation.  
- **lookback_ratio**: Ratio of attention between input and output tokens (used for comparison).  
  - Reference: [Lookback-Lens](https://github.com/voidism/Lookback-Lens)  
  - Also uses `gen_features.py`.

If using LLaMA models, you must apply for access on Hugging Face and obtain a token.

To change the storage location of the LLM, modify the following:

```python
# Change Hugging Face cache directory (if needed)
os.environ['HF_HOME'] = "/home/code/llm"
```

- Features are saved in `/home/code/data/saves/{model_type}_{task_type}.pkl`  
- Each file may be **50–80 GB**, so storage space should be carefully managed.  
- **Tested and confirmed to run.**  
- `matrix_device_{o,v}`: GPU devices for reshaping attention.

---

### `2_ragtruth_preprocessing.ipynb`
Converts features created in `1_create_data.ipynb` into a format usable for Transformer encoder training (**RAGTruth version**).

- Requires that features have been saved to `/home/code/data/saves/{model_type}_{task_type}.pkl`.  
- Performs hallucination labeling **per token**, and splits data into training, validation, and test sets.  
- Split files are also large in size.  

---

### `2_truthqa_preprocessing.ipynb`
Converts features from `1_create_data.ipynb` for **TruthfulQA sentence-level hallucination detection**.

- Includes fine-tuning of a labeling model (`gpt-4o-mini`).  
- Note: Newer models may have clearly better performance—please update as needed.  
- Labels each sentence in TruthfulQA as hallucinated or not.  
- Splits data into train/val/test with a balanced number of hallucinated and non-hallucinated sentences.


---

### `3_training.ipynb`
Trains and evaluates a Transformer encoder using attention-based features.

- Works with:
  - `optimize.py`: Training logic.
  - `evaluate.py`: Evaluation logic.
    - Provides evaluation metrics and output examples to manually verify which hallucinations are detected.

- Loading `.pkl` feature files can take significant time. Intermediate preprocessed files are saved to:

```
/home/code/data/cashe_file/features/{model_type}/preprocessed_data_{model_type}{dataset_name}_{feature_type}_all_features.pt
```

> ⚠️ These intermediate files are also large.

- First-time execution may take **up to 1 hour** before training starts due to preprocessing.
- Training itself is time-consuming:
  - Up to 150 epochs and 200 trials.
  - **Uses 2 GPUs**, taking around **2–4 days**.
  - The code is optimized based on the number of GPUs used.

---

    #### Important Parameters
    
    - **mode**
      - `span`: Token-level hallucination detection (for QA, Data2txt, Summary)  
      - `pooling`: Sentence-level hallucination detection (for TruthfulQA)
    
    - **span_decoder**
      - Chooses between CRF and linear layer.  
      - **CRF is recommended.**
    
    - **feature_type**
      - `raw`: Uses raw attention values.  
      - `norm`: Uses attention multiplied by norms (as in Kobayashi et al.).
    
    - **features_to_use**
      - Select from: `key_avg`, `query_entropy`, `key_entropy`, `lookback_ratio`.  
      - Use comma-separated values for multiple features (e.g., `"key_avg,query_entropy"`).
    
    - **layers**
      - Specify attention layer indices as a comma-separated list.  
      - For all layers:
        - LLaMA: `0–31`  
        - Qwen: `0–27`
    
    - **heads**
      - Specify attention head indices, in the same format as `layers`.
    
    - **clf_mode**
      - Typically set to `transformer`.  
      - `lookback_ratio_lr`: Used for applying logistic regression on the `lookback_ratio` feature for comparison.
    
    - **pooling_type**
      - Specific to **TruthfulQA** mode; refer to the master’s thesis for details.
    
    - **top_models_json_path**
      - Path to a JSON file (within `training_file`) that contains hyperparameters from previous training sessions for reuse.
    

- その他
    - llamaやqwenのfine-tuningはhttps://github.com/hiyouga/LLaMA-Factoryを使って行っています．
    - 
