# Hallucination Detection using Multi-View Attention Features

## Research Overview

This research focuses on detecting hallucinations in Large Language Models (LLMs) at the **token level**.

- A Transformer encoder-based model is trained via **supervised learning** to predict whether each token contains hallucination.

- **RAGTruth** is used as the labeled dataset for training and evaluation.

- Additionally, experiments are conducted under a **sentence-level hallucination detection** setting using **TruthfulQA**.





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

    
    

- メインのコードの簡単な説明 (/featuresに含まれる)
    - 1_create_data.ipynb
        - アテンションに関する特徴量を作成
            - 各トークンが，他のトークンから受けた注目度の平均
                - key_avg
            - 各トークンに，注目を向けているトークンの多様さ
                - key_entropy
            - 各トークンの生成時に，注目を与えたトークンの多様さ
                - query_entropy
            - 比較手法の，入力と出力にかかるアテンション比
                - lookback_ratio
                - 参考 https://github.com/voidism/Lookback-Lens
        - 同時にgen_features.pyも使用
        - llamaのモデルを使う場合は，hugging faceに申請を行い，トークンを発行する必要がある
        - 以下のパスを変更することで，llmの保存先を変更することが可能
            
            ```
            #huggingfaceのcacheを変更 (必要ならば)
            os.environ['HF_HOME'] = "/home/code/vishnu/llm"
            ```
            
        - /home/code/data/saves/{model_type}_{task_type}.pklに特徴量が保存される
            - 1ファイル辺り50-80GBくらいの容量の大きなファイルになるので注意
        - 動作確認済
        - matrix_device_{o,v}：アテンション変形用のGPU
    - 2_ragtruth_preprocessing.ipynb
        - 1のコードで作った**ragtruth用の**特徴量をtransformerエンコーダの学習に使用できる形式に変換 (ragtruthバージョン)
            - /home/code/data/saves/{model_type}_{task_type}.pkl　に 1_create_dataで作成した特徴量を保存する必要がある
            - このコードによって，トークン辺りのハルシネーションラベル付け，訓練・検証・テストデータへの分割などが行われる
                - 分割後のファイルも容量が大きいので，注意
        - 動作確認済
    - 2_truthqa_preprocessing.ipynb
        - 1のコードで作ったtruthfulqa**用の**特徴量をtransformerエンコーダの学習に使用できる形式に変換 (truthqaバージョン)
            - ラベル付け用モデルgpt-4o-miniのfine-tuning
                - 現状はこれより明らかに性能が良いモデルがあると思うので，適宜更新してください
            - そのモデルを使った，truthfulqaの文単位でのラベル付け
                - hallucinationが発生している文と発生していない文が訓練データ，検証データ，テストデータにおいて均等になるように分割
        - モデルのfine-tuningの箇所など一部動作確認していない箇所があります
            - もしエラーが出たらすみません
    - 3_training.ipynb
        - アテンションに関する特徴量を使ったtransformerエンコーダの学習と評価
        - optimize.pyとevaluate.pyと併用
            - optimize.pyはモデルの訓練用コード
            - evaluate.pyはモデルの評価用コード
                - 評価精度と，具体的な出力が分かるので，どのハルシネーションを検出できているかなどの人目での評価が別プログラムによって可能になります．
        - 特徴量をpklファイルから呼び出す際に非常に時間がかかるので，中間ファイル (/home/code/data/cashe_file/features/{model_type}/preprocessed_data_{model_type}_{dataset_name}*_*{feature_type}_all_features.pt) 
        にデータを保存
            - このファイルも容量が大きいので注意
            - 初回実行時は中間ファイルがないので，学習開始までにかなり時間がかかります
                - 1時間程度
        - trainingにはかなり時間がかかります
            - 150epoch max， 200trial，GPU2台使用で2~4日くらいかかる
        - 1回の実行には，GPUを2台使うのが良いと思います
            - GPUの使用台数によって，プログラムは最適化するようにしています
        - パラメータについて
            - mode
                - span
                    - トークン単位のハルシネーション検出
                        - QA, Data2txt, Summary用
                - pooling
                    - 文単位のハルシネーション検出
                        - truthqa用
            - span_decoder
                - crf層を使うか，linear層を使うか
                - 基本はcrfでOK
            - feature_type
                - raw
                    - アテンションをそのまま使う
                - norm
                    - Kobayashi et al.のノルムをアテンションに掛け算する手法を使って取得した特徴量
            - features_to_use
                - key_avg,query_entropy,key_entropy,lookback_ratio の中から選択．複数選択する場合はカンマ区切りで指定．例: "key_avg,query_entropy”
            - layers
                - 使用するアテンションの層の番号をすべてカンマ区切りで指定
                    - 全ての層を使いたい場合は，llamaなら0~31
                    - qwenなら0~27をカンマ区切りで指定
            - heads
                - 使用するアテンションヘッドの番号をすべてカンマ区切りで指定
                - layersと基本指定方法は同じ
            - clf_mode
                - 基本はtransformerでOK
                - lookback_ratio_lrは比較手法のロジスティック回帰をlookback_ratioの特徴量に対して試したいとき専用
            - pooling_type
                - truthfulqa専用モード，修論参照
            - top_models_json_path
                - 過去に学習したモデルのハイパラを使いまわす用のコード
                - training_fileに含まれるjsonのpathを指定

- 評価用コードの簡単な説明 (eval_file)
    - 一部修正しましたが，pathなど合っていない箇所があるかもしれないです，すみません
        - 適宜使用する際修正してください
    - confirmation_hallcination_examples.ipynb
        - 具体的にどこがハルシネーションしているか図示するコード
    - eval_finetuning_model.ipynb
        - 比較手法のfine-tuningしたllamaやqwen等の性能を評価
    - eval_per_hal_type_token_count.ipynb
        - hallucinationを引き起こしているtokenのtype別調査
    - eval_per_model_token_count.ipynb
        - RAGTruthには複数のモデルの出力が含まれる
            - そのhallucinationをどのモデルによって出力されているかを評価
    - ragtruth_token_num_count.ipynb
        - ragtruthのデータセットに含まれるハルシネーションのトークン数をカウント
    - visualization_attention.ipynb
        - LLMの各文に対するattentionの大きさを可視化する
            - llama3_gen.pyと併用
    

- 大規模データ
    - 過去に使用したアテンションの特徴量に関するデータや，モデルのハイパラに関するデータを保存しています
        - 特徴量に関するデータは，2_ragtruth_preprocessing.ipynbまで実行したものになっています
    - training_fileのmainフォルダに含まれるものは，主要な実験の結果です
        - こちらはpathやファイル名などもそのまま使えるように修正しました
            - 環境の違いなどでもし，使えなかった場合はすみません，適宜修正してください
    - subフォルダに含まれるものは主要ではないサブ実験です
        - pathは修正しましたが，ファイル名などが非常に適当になっています，すみません.適宜修正してください
            - linear_syuron
                - 修論に含まれる線形層を使った実験
            - logistic
                - lookback_ratio_lrを使用した評価
            - qwen_attention_only
            - truthqa
                - truthfulqaに関する実験
            - use_all_features_syuron
                - lookback_ratioも含めた4種類の特徴量のもの
            
            attention_only →　oneに対応
            
            without_lookback → allに対応
            
            sub/use_all_features_syuron → lookback_ratioも含めた4種類の特徴量のもの
            
    
    ギガファイル便
    
    passwordはすべて　0317　です。
    
    training_file.zip
    
    [https://93.gigafile.nu/0624-m50b17b9c1f8972c8c8f5494d45e508db](https://93.gigafile.nu/0624-m50b17b9c1f8972c8c8f5494d45e508db)
    
    231.14GBsaves_llama.zip
    
    [https://93.gigafile.nu/0624-f6a299fd5195931b34cb5ec2c20f6b020](https://93.gigafile.nu/0624-f6a299fd5195931b34cb5ec2c20f6b020)
    
    152.42GBsaves_qwen.zip
    
    [https://93.gigafile.nu/0624-p5ca16328e3b8bb2240c6f24c033fe674](https://93.gigafile.nu/0624-p5ca16328e3b8bb2240c6f24c033fe674)
    
    114.1GB
    
    上3つをまとめたもの
    
    ogasa_hal_data.zip
    
    [https://93.gigafile.nu/0624-cc954553cd63549ace819f023c88053a](https://93.gigafile.nu/0624-cc954553cd63549ace819f023c88053a)
    
    [https://xgf.nu/H7sZi](https://xgf.nu/H7sZi)
    
    497.67GB
    

- その他
    - llamaやqwenのfine-tuningはhttps://github.com/hiyouga/LLaMA-Factoryを使って行っています．
    - なにか分からないことがあればslackか
    [ogasa.yu.9270@outlook.jp](mailto:何かわからないことがあれば，slackかogasa.yu.9270@outlook.jp) までお願いします
