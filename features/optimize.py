import argparse
import logging
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import time
import pandas as pd
from torchcrf import CRF  # ★ Used in the first code (for CRF)
import copy
import pickle
from sklearn.linear_model import LogisticRegression  # ★ Added (using sklearn's logistic regression)
import gc  # ★ Added (for garbage collection)


def main():
    """
    This script integrates two codes so that in span mode, you can switch between CRF or Linear
    using --span_decoder.

    The content of the first code (CRF in span mode) and
    the content of the second code (linear in span mode) are integrated,
    and by adding the argument --span_decoder (crf or linear),
    it is possible to switch between either process.

    It includes all processes that existed only in one of the two codes without omission.
    """
    parser = argparse.ArgumentParser(description="A script that selects arbitrary features and layer, head regions from all preprocessed feature data, optimizes using Transformer, performs CRF/linear classification in pooling mode or span mode, and optionally logistic regression.")
    parser.add_argument('--mode', type=str, required=True, choices=['pooling', 'span'],
                        help="Mode: 'pooling' or 'span'")
    parser.add_argument('--dataset_name', type=str, required=True,
                        help="Name of the dataset")
    parser.add_argument('--feature_type', type=str, required=True, choices=['raw', 'norm'],
                        help="Feature type: 'raw' or 'norm'")
    parser.add_argument('--save_name', type=str, default='default_save_name',
                        help="Save file name")
    parser.add_argument('--pooling_type', type=str, default='mean',
                        help="Pooling type: 'any', 'count', 'mean', 'max', 'cls', 'attention'")
    parser.add_argument('--train_path', type=str, required=True,
                        help="Training data path (pickle file)")
    parser.add_argument('--val_path', type=str, required=True,
                        help="Validation data path (pickle file)")
    parser.add_argument('--test_path', type=str, required=True,
                        help="Test data path (pickle file)")
    parser.add_argument('--n_epochs', type=int, default=50,
                        help="Number of epochs (for Transformers)")
    parser.add_argument('--n_trials', type=int, default=50,
                        help="Number of Optuna trials (for Transformers)")
    parser.add_argument('--batch_size', type=int, required=True,
                        help="Batch size")
    parser.add_argument('--features_to_use', type=str, default="key_avg,query_entropy,key_entropy,lookback_ratio",
                        help="Features to use (comma-separated). Extracted from all features.")
    parser.add_argument('--layers_to_use', type=str, default="0-31",
                        help="Layer numbers to use. Example: '0-31' or '0,1,2', etc.")
    parser.add_argument('--heads_to_use', type=str, default="0-31",
                        help="Head numbers to use. Example: '0-31' or '0,1,2', etc.")
    parser.add_argument('--clf_mode', type=str, default='transformer',
                        help="Classification mode: 'transformer' or 'lookback_ratio_lr'")
    parser.add_argument('--top_models_json_path', type=str, default=None,
                        help="Path to the JSON file of the top 5 trials previously output. If specified, Optuna is not run, and only that parameter set is used for training.")
    # ★ Added an option to switch between CRF or Linear in span mode
    parser.add_argument('--span_decoder', type=str, default='crf', choices=['crf', 'linear'],
                        help="Decoder type in span mode: 'crf' or 'linear'")
    
    # Choose the type of model
    parser.add_argument('--model_type', type=str, default='llama', choices=['llama', 'qwen'],
                        help="Type of model to retrieve features: 'llama' or 'qwen'")

    args = parser.parse_args()

    mode = args.mode
    dataset_name = args.dataset_name
    feature_type = args.feature_type
    save_name = args.save_name
    pooling_type = args.pooling_type if mode == 'pooling' else None
    train_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path
    n_epochs = args.n_epochs
    n_trials = args.n_trials
    batch_size = args.batch_size
    requested_features = [f.strip() for f in args.features_to_use.split(',') if f.strip()]
    layers_to_use_str = args.layers_to_use
    heads_to_use_str = args.heads_to_use
    clf_mode = args.clf_mode
    top_models_json_path = args.top_models_json_path
    span_decoder = args.span_decoder  # ★ CRF or Linear
    model_type = args.model_type 

    def parse_range(range_str):
        if '-' in range_str:
            start, end = range_str.split('-')
            return list(range(int(start), int(end) + 1))
        else:
            return [int(x) for x in range_str.split(',')]

    layers_to_use = parse_range(layers_to_use_str)
    heads_to_use = parse_range(heads_to_use_str)

    all_possible_features = ['key_avg', 'query_entropy', 'key_entropy', 'lookback_ratio']
    chosen_feature_names = [f"{feature_type}_{feat}" for feat in all_possible_features]

    output_dir = f"/home/code/data/training_file/{save_name}_{feature_type}"
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    log_filename = (
        f"transformer_optimize_{save_name}_{pooling_type}_{feature_type}.log"
        if mode == 'pooling'
        else f"transformer_optimize_{save_name}_{feature_type}.log"
    )
    log_file = os.path.join(output_dir, log_filename)
    logger = logging.getLogger("transformer_optimize_logger")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("----- Execution Parameters -----")
    logger.info(f"Mode: {mode}")
    logger.info(f"Dataset Name: {dataset_name}")
    logger.info(f"Feature Type: {feature_type}")
    logger.info(f"Save Name: {save_name}")
    if mode == 'pooling':
        logger.info(f"Pooling Type: {pooling_type}")
    logger.info(f"Train Path: {train_path}")
    logger.info(f"Validation Path: {val_path}")
    logger.info(f"Test Path: {test_path}")
    logger.info(f"Number of Epochs: {n_epochs}")
    logger.info(f"Number of Trials: {n_trials}")
    logger.info(f"Batch Size: {batch_size}")
    logger.info(f"Requested Features: {requested_features}")
    logger.info(f"Layers to use: {layers_to_use}")
    logger.info(f"Heads to use: {heads_to_use}")
    logger.info(f"Classification Mode: {clf_mode}")
    logger.info(f"Top Models JSON Path: {top_models_json_path}")
    logger.info(f"Span Decoder: {span_decoder}")
    logger.info("---------------------------")

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    set_seed()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    num_available_gpus = torch.cuda.device_count()
    if num_available_gpus > 1:
        logger.info(f"Number of available GPUs: {num_available_gpus}. Using DataParallel.")
    else:
        logger.info(f"Number of available GPUs: {num_available_gpus}. Using single GPU or CPU.")

    class EarlyStopping:
        def __init__(self, patience=10, verbose=False, delta=0.01):
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.best_epoch = 0
            self.delta = delta
            self.best_model_state = None

        def __call__(self, val_f1, current_epoch, model):
            score = val_f1
            if self.best_score is None:
                self.best_score = score
                self.best_epoch = current_epoch
                self.best_model_state = self._get_model_state(model)
                self.counter = 0
                if self.verbose:
                    logger.info(f"Initial score: {score:.4f} at epoch {current_epoch}")
            elif score >= self.best_score + self.delta:
                improvement = score - self.best_score
                self.best_score = score
                self.best_epoch = current_epoch
                self.best_model_state = self._get_model_state(model)
                self.counter = 0
                if self.verbose:
                    logger.info(f"Score updated: {score:.4f} (Improvement: +{improvement:.4f}) at epoch {current_epoch}")
            else:
                self.counter += 1
                if self.verbose:
                    logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
                    if self.verbose:
                        logger.info("Early stopping triggered")

        def _get_model_state(self, model):
            if isinstance(model, nn.DataParallel):
                return copy.deepcopy(model.module.state_dict())
            else:
                return copy.deepcopy(model.state_dict())

    # ★ Common folder name for data used in the first/second code
    preprocessed_dir = f"/home/code/data/cashe_file/features/{model_type}"
    os.makedirs(preprocessed_dir, exist_ok=True)
    preprocessed_file = f'preprocessed_data_{model_type}_{dataset_name}_{feature_type}_all_features.pt'
    preprocessed_path = os.path.join(preprocessed_dir, preprocessed_file)

    def load_and_preprocess_data(feature_type, pooling_type=None, train_path=None, val_path=None, test_path=None):
        logger.info("Starting data loading and preprocessing")
        if os.path.exists(preprocessed_path):
            logger.info(f"Detected preprocessed file {preprocessed_path}. Loading it.")
            preprocessed_data = torch.load(preprocessed_path)
            train_dataset = preprocessed_data['train_dataset']
            val_dataset = preprocessed_data['val_dataset']
            test_dataset = preprocessed_data['test_dataset']
            input_dim = preprocessed_data['input_dim']
            max_length = preprocessed_data['max_length'] if mode == 'span' else None
            train_all_texts = preprocessed_data.get('train_all_texts', [])
            val_all_texts = preprocessed_data.get('val_all_texts', [])
            test_all_texts = preprocessed_data.get('test_all_texts', [])

            if mode == 'pooling':
                return train_dataset, val_dataset, test_dataset, input_dim
            else:
                return (
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    input_dim,
                    max_length,
                    train_all_texts,
                    val_all_texts,
                    test_all_texts,
                )
        else:
            logger.info(f"No preprocessed file {preprocessed_path} found. Starting processing.")

            train_df = pd.read_pickle(train_path)
            val_df = pd.read_pickle(val_path)
            test_df = pd.read_pickle(test_path)

            if mode == 'span':
                max_length = 0
                train_df_labels = train_df['all_texts_hal_token_label'].dropna()
                for labels_seq in train_df_labels:
                    valid_length = sum([1 for label in labels_seq if label != -1])
                    max_length = max(max_length, valid_length)
                val_df_labels = val_df['all_texts_hal_token_label'].dropna()
                for labels_seq in val_df_labels:
                    valid_length = sum([1 for label in labels_seq if label != -1])
                    max_length = max(max_length, valid_length)
                test_df_labels = test_df['all_texts_hal_token_label'].dropna()
                for labels_seq in test_df_labels:
                    valid_length = sum([1 for label in labels_seq if label != -1])
                    max_length = max(max_length, valid_length)
                logger.info(f"max_length={max_length} calculation complete")

            # Both for llama and qwen (32->28, 1024->784)
            def preprocess_span(df, max_length, model_type):
                feature_blocks = {feat: [] for feat in chosen_feature_names}
                labels_list = []
                labels_mask_list = []
                attention_masks_list = []

                for i in range(len(df)):
                    labels = df.iloc[i]['all_texts_hal_token_label']
                    if labels is None or len(labels) == 0:
                        continue
                    label_sequence = torch.tensor(labels, dtype=torch.float32)
                    valid_indices = (label_sequence != -1).nonzero(as_tuple=False).view(-1)
                    if valid_indices.numel() == 0:
                        continue

                    valid_labels = label_sequence[valid_indices]
                    current_length = valid_labels.shape[0]
                    if current_length > max_length:
                        valid_labels = valid_labels[:max_length]
                        current_length = max_length

                    pad_size = max_length - current_length
                    if pad_size > 0:
                        label_padding = torch.full((pad_size,), -1.0, dtype=torch.float32)
                        valid_labels = torch.cat((valid_labels, label_padding), dim=0)

                    labels_list.append(valid_labels.unsqueeze(0))

                    label_mask = (valid_labels != -1).float()
                    labels_mask_list.append(label_mask.unsqueeze(0))
                    attention_mask = (valid_labels == -1)
                    attention_masks_list.append(attention_mask.unsqueeze(0))

                    for feature_name in chosen_feature_names:
                        feature = df.iloc[i][feature_name]
                        valid_feature = feature[:, :, valid_indices]
                        if current_length > max_length:
                            valid_feature = valid_feature[:, :, :max_length]
                        if pad_size > 0:
                            if model_type == 'llama':
                                feature_padding = torch.zeros(32, 32, pad_size)
                            elif model_type == 'qwen':
                                feature_padding = torch.zeros(28, 28, pad_size)
                            valid_feature = torch.cat((valid_feature, feature_padding), dim=-1)
                        elif pad_size < 0:
                            valid_feature = valid_feature[:, :, :max_length]
                        
                        if model_type == 'llama':
                            reshaped = valid_feature.permute(2, 0, 1).reshape(max_length, 32 * 32)
                        elif model_type == 'qwen':
                            reshaped = valid_feature.permute(2, 0, 1).reshape(max_length, 28 * 28)
                        feature_blocks[feature_name].append(reshaped.unsqueeze(0))

                if len(labels_list) == 0:
                    raise ValueError("No valid data was found.")

                labels_tensor = torch.cat(labels_list, dim=0).long()
                labels_mask_tensor = torch.cat(labels_mask_list, dim=0)
                masks_tensor = torch.cat(attention_masks_list, dim=0)

                normalized_feature_tensors = []
                for feature_name in chosen_feature_names:
                    blocks = torch.cat(feature_blocks[feature_name], dim=0)
                    if 'lookback_ratio' in feature_name:
                        normalized_blocks = blocks
                    else:
                        mean = blocks.mean(dim=(0, 1))
                        std = blocks.std(dim=(0, 1))
                        std[std == 0] = 1.0
                        normalized_blocks = (blocks - mean) / std
                    normalized_feature_tensors.append(normalized_blocks)

                features_tensor = torch.cat(normalized_feature_tensors, dim=-1)
                input_dim = features_tensor.shape[2]

                dataset = TensorDataset(features_tensor, labels_tensor, masks_tensor, labels_mask_tensor)
                return dataset, input_dim
            
            # Both for llama and qwen (32->28, 1024->784)
            def preprocess_pooling(df, pooling_type, model_type):
                labels = df['label'].tolist()
                features_dict = {
                    feature_name: df[feature_name].tolist() for feature_name in chosen_feature_names
                }
                lengths = [f.shape[-1] for f in features_dict[chosen_feature_names[0]]]
                max_length = max(lengths) + 30
                num_samples = len(labels)

                normalized_features_list = []
                for feature_name in chosen_feature_names:
                    features = features_dict[feature_name]
                    padded_features = []
                    for f_ in features:
                        seq_length = f_.shape[-1]
                        padding_length = max_length - seq_length
                        if padding_length > 0:
                            if model_type == 'llama':
                                padding = torch.zeros((32, 32, padding_length))
                            elif model_type == 'qwen':
                                padding = torch.zeros((28, 28, padding_length))
                            feature_padded = torch.cat((f_, padding), dim=-1)
                        else:
                            feature_padded = f_[:, :, :max_length]
                        padded_features.append(feature_padded.unsqueeze(0))

                    features_tensor = torch.cat(padded_features, dim=0)
                    num_samples_ = features_tensor.shape[0]
                    seq_length_ = features_tensor.shape[-1]
                    if model_type == 'llama':
                        input_dim_single = 32 * 32
                    elif model_type == 'qwen':
                        input_dim_single = 28 * 28
                    features_tensor = features_tensor.permute(0, 3, 1, 2).reshape(
                        num_samples_, seq_length_, input_dim_single
                    )
                    if 'lookback_ratio' in feature_name:
                        pass
                    else:
                        mean = features_tensor.mean(dim=(0, 1))
                        std = features_tensor.std(dim=(0, 1))
                        std[std == 0] = 1.0
                        features_tensor = (features_tensor - mean) / std
                    normalized_features_list.append(features_tensor)

                combined_features_tensor = torch.cat(normalized_features_list, dim=-1)
                input_dim = combined_features_tensor.shape[-1]

                if pooling_type in ['any', 'count']:
                    masks_tensor = (combined_features_tensor.sum(dim=-1) == 0)
                else:
                    masks_tensor = torch.zeros((num_samples, max_length), dtype=torch.bool)

                labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
                dataset = TensorDataset(combined_features_tensor, labels_tensor, masks_tensor)
                return dataset, input_dim, max_length

            if mode == 'pooling':
                train_dataset, input_dim, max_length_train = preprocess_pooling(train_df, pooling_type)
                val_dataset, input_dim_val, max_length_val = preprocess_pooling(val_df, pooling_type)
                test_dataset, input_dim_test, max_length_test = preprocess_pooling(test_df, pooling_type)
                assert input_dim == input_dim_val == input_dim_test
                max_length = max_length_train
                train_all_texts, val_all_texts, test_all_texts = [], [], []
            elif mode == 'span':
                train_dataset, input_dim = preprocess_span(train_df, max_length,model_type)
                val_dataset, input_dim_val = preprocess_span(val_df, max_length,model_type)
                test_dataset, input_dim_test = preprocess_span(test_df, max_length,model_type)
                assert input_dim == input_dim_val == input_dim_test

                train_all_texts = (
                    train_df['full_prompt'].tolist()
                    if 'full_prompt' in train_df.columns
                    else [""] * len(train_dataset)
                )
                val_all_texts = (
                    val_df['full_prompt'].tolist()
                    if 'full_prompt' in val_df.columns
                    else [""] * len(val_dataset)
                )
                test_all_texts = (
                    test_df['full_prompt'].tolist()
                    if 'full_prompt' in test_df.columns
                    else [""] * len(test_dataset)
                )
            else:
                raise ValueError("mode must be 'pooling' or 'span'")

            save_dict = {
                'train_dataset': train_dataset,
                'val_dataset': val_dataset,
                'test_dataset': test_dataset,
                'input_dim': input_dim,
            }
            if mode == 'span':
                save_dict['max_length'] = max_length
                save_dict['train_all_texts'] = train_all_texts
                save_dict['val_all_texts'] = val_all_texts
                save_dict['test_all_texts'] = test_all_texts

            torch.save(save_dict, preprocessed_path)
            logger.info(f"Saved preprocessed data to {preprocessed_path}.")

            if mode == 'pooling':
                return train_dataset, val_dataset, test_dataset, input_dim
            else:
                return (
                    train_dataset,
                    val_dataset,
                    test_dataset,
                    input_dim,
                    max_length,
                    train_all_texts,
                    val_all_texts,
                    test_all_texts,
                )

    try:
        if mode == 'pooling':
            train_dataset, val_dataset, test_dataset, input_dim = load_and_preprocess_data(
                feature_type,
                pooling_type=pooling_type,
                train_path=train_path,
                val_path=val_path,
                test_path=test_path
            )
        elif mode == 'span':
            (
                train_dataset,
                val_dataset,
                test_dataset,
                input_dim,
                max_length,
                train_all_texts,
                val_all_texts,
                test_all_texts,
            ) = load_and_preprocess_data(
                feature_type,
                pooling_type=None,
                train_path=train_path,
                val_path=val_path,
                test_path=test_path
            )

        logger.info("Data loading/preprocessing complete")

        # Feature re-extraction for (layer, head, feature_type)
        # Both for llama and qwen (32->28, 1024->784)
        def subset_features_dataset(dataset, mode,model_type):
            all_tensors = dataset.tensors
            features = all_tensors[0]
            selected_features_list = []
            for i, feat_name in enumerate(chosen_feature_names):
                base_name = feat_name.replace(f"{feature_type}_", "")
                if base_name in requested_features:
                    if model_type == 'llama':
                        start = i * 1024
                        end = (i + 1) * 1024
                        feature_chunk = features[:, :, start:end]
                        feature_chunk = feature_chunk.view(feature_chunk.size(0), feature_chunk.size(1), 32, 32)
                        feature_chunk = feature_chunk[:, :, layers_to_use, :]
                        feature_chunk = feature_chunk[:, :, :, heads_to_use]
                        feature_chunk = feature_chunk.view(feature_chunk.size(0), feature_chunk.size(1), -1)
                        selected_features_list.append(feature_chunk)
                
                elif model_type == 'qwen':
                    start = i * 784
                    end = (i + 1) * 784
                    feature_chunk = features[:, :, start:end]
                    feature_chunk = feature_chunk.view(feature_chunk.size(0), feature_chunk.size(1), 28, 28)
                    feature_chunk = feature_chunk[:, :, layers_to_use, :]
                    feature_chunk = feature_chunk[:, :, :, heads_to_use]
                    feature_chunk = feature_chunk.view(feature_chunk.size(0), feature_chunk.size(1), -1)
                    selected_features_list.append(feature_chunk)
                    

            new_features = torch.cat(selected_features_list, dim=-1) if selected_features_list else features

            if mode == 'pooling':
                labels = all_tensors[1]
                masks = all_tensors[2]
                return TensorDataset(new_features, labels, masks)
            else:
                labels = all_tensors[1]
                masks = all_tensors[2]
                labels_mask = all_tensors[3]
                return TensorDataset(new_features, labels, masks, labels_mask)

        train_dataset = subset_features_dataset(train_dataset, mode, model_type)
        val_dataset = subset_features_dataset(val_dataset, mode, model_type)
        test_dataset = subset_features_dataset(test_dataset, mode, model_type)

        new_input_dim = len(requested_features) * (len(layers_to_use) * len(heads_to_use))

        # ★ Common to the first code/second code: "lookback_ratio_lr" (logistic regression)
        #   In span mode, if the features are only lookback_ratio, train with a window
        if (
            clf_mode == 'lookback_ratio_lr'
            and mode == 'span'
            and requested_features == ['lookback_ratio']
        ):
            sliding_window_sizes = [1, 8]
            train_val_features = torch.cat([train_dataset.tensors[0], val_dataset.tensors[0]], dim=0)
            train_val_labels = torch.cat([train_dataset.tensors[1], val_dataset.tensors[1]], dim=0)
            train_val_masks = torch.cat([train_dataset.tensors[2], val_dataset.tensors[2]], dim=0)
            train_val_labels_mask = torch.cat([train_dataset.tensors[3], val_dataset.tensors[3]], dim=0)
            test_features, test_labels, test_masks, test_labels_mask = test_dataset.tensors

            # --- Modified point here ---
            # Where (y_full != 0) & (y_full != 1) & mask_full was changed to
            # ((y_full == 0) | (y_full == 1)) & mask_full
            def sliding_window(features, labels, labels_mask, window_size):
                N, seq_len, dim = features.shape
                windows = features.unfold(1, window_size, 1)
                X = windows.contiguous().view(N, -1, window_size * dim)

                y_full = labels[:, window_size - 1:]
                mask_full = labels_mask[:, window_size - 1:].bool()

                # ★ Modified: Take positions where labels are 0 or 1 and mask_full is True
                valid_label_mask = ((y_full == 0) | (y_full == 1)) & mask_full

                valid_mask_flat = valid_label_mask.reshape(-1)
                y_filtered = y_full.reshape(-1)[valid_mask_flat]
                X_filtered = X.reshape(-1, window_size * dim)[valid_mask_flat]

                if torch.any((y_filtered != 0) & (y_filtered != 1)):
                    raise ValueError("Labels contain values other than 0 or 1.")

                return X_filtered.cpu().numpy(), y_filtered.cpu().numpy()
            # --- Modification ends here ---

            for window_size in sliding_window_sizes:
                logger.info(f"----- Logistic regression classification: window size={window_size} -----")
                print(f"----- Logistic regression classification: window size={window_size} -----")

                try:
                    X_train, y_train = sliding_window(
                        train_val_features, train_val_labels, train_val_labels_mask, window_size=window_size
                    )
                    X_test, y_test = sliding_window(
                        test_features, test_labels, test_labels_mask, window_size=window_size
                    )
                except Exception as e:
                    logger.error(f"Error occurred in sliding window processing with window size {window_size}: {e}", exc_info=True)
                    print(f"Error occurred in sliding window processing with window size {window_size}: {e}")
                    continue

                if len(X_train) == 0 or len(X_test) == 0:
                    logger.warning(f"No valid samples found at window size {window_size}. Skipping.")
                    print(f"No valid samples found at window size {window_size}. Skipping.")
                    continue

                logger.info(f"Train windows: {X_train.shape[0]}, Test windows: {X_test.shape[0]}")
                print(f"Train windows: {X_train.shape[0]}, Test windows: {X_test.shape[0]}")

                lr_model = LogisticRegression(max_iter=1000)
                lr_model.fit(X_train, y_train)

                y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
                y_pred = (y_pred_proba >= 0.5).astype(int)

                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                conf_mat = confusion_matrix(y_test, y_pred)
                try:
                    auroc = roc_auc_score(y_test, y_pred_proba)
                except ValueError:
                    auroc = float('nan')

                logger.info(f"Logistic regression evaluation result (window size {window_size}):")
                logger.info(f"Precision: {precision:.4f}")
                logger.info(f"Recall: {recall:.4f}")
                logger.info(f"F1-score: {f1:.4f}")
                logger.info(f"Confusion Matrix:\n{conf_mat}")
                logger.info(f"AUROC: {auroc:.4f}")

                print(f"Logistic regression evaluation result (window size {window_size}):")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1-score: {f1:.4f}")
                print("Confusion Matrix:")
                print(conf_mat)
                print(f"AUROC: {auroc:.4f}")

                model_filename = f"lr_model_{dataset_name}_{save_name}_{feature_type}_lookback_ratio_span_window_{window_size}.pkl"
                model_path = os.path.join(models_dir, model_filename)

                with open(model_path, "wb") as f:
                    pickle.dump(lr_model, f)

                logger.info(f"Saved logistic regression model (window size {window_size}) to {model_path}.")
                print(f"Saved logistic regression model (window size {window_size}) to {model_path}.")

        else:
            # ★ Transformer part starts here

            class CustomTransformerEncoderLayer(nn.Module):
                def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
                    super(CustomTransformerEncoderLayer, self).__init__()
                    self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
                    self.linear1 = nn.Linear(d_model, dim_feedforward)
                    self.dropout = nn.Dropout(dropout)
                    self.linear2 = nn.Linear(dim_feedforward, d_model)
                    self.norm1 = nn.LayerNorm(d_model)
                    self.norm2 = nn.LayerNorm(d_model)
                    self.dropout1 = nn.Dropout(dropout)
                    self.dropout2 = nn.Dropout(dropout)

                def forward(self, src, src_mask=None, src_key_padding_mask=None):
                    src2, attn_weights = self.self_attn(
                        src, src, src,
                        attn_mask=src_mask,
                        key_padding_mask=src_key_padding_mask,
                        need_weights=True,
                        average_attn_weights=False
                    )
                    src = src + self.dropout1(src2)
                    src = self.norm1(src)
                    src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
                    src = src + self.dropout2(src2)
                    src = self.norm2(src)
                    return src, attn_weights

            class TransformerClassifier(nn.Module):
                """
                mode='pooling': Classify token sequences by pooling to get a binary classification (1-dimensional output)
                mode='span':   Classify each token in the token sequence as binary, then
                               decode using CRF or simply predict 0/1 (linear).

                span_decoder='crf' or 'linear' switches between these.
                """
                def __init__(
                    self,
                    input_dim,
                    model_dim=1024,
                    num_heads=4,
                    num_layers=4,
                    max_length=300,
                    dropout_prob=0.3,
                    pooling_type=None,
                    mode='pooling',
                    span_decoder='crf'
                ):
                    super(TransformerClassifier, self).__init__()
                    self.input_fc = nn.Linear(input_dim, model_dim)
                    self.mode = mode
                    self.pooling_type = pooling_type
                    self.dropout = nn.Dropout(dropout_prob)
                    self.positional_encoding = self._generate_positional_encoding(
                        max_length + (1 if pooling_type else 0), model_dim
                    )
                    self.register_buffer('positional_encoding_buffer', self.positional_encoding.unsqueeze(0))
                    self.layers = nn.ModuleList([
                        CustomTransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout_prob)
                        for _ in range(num_layers)
                    ])
                    self.span_decoder = span_decoder

                    if self.mode == 'pooling':
                        # ★ For pooling: binary classification (0 or 1) => 1-dimensional output
                        self.fc_out = nn.Linear(model_dim, 1)

                    elif self.mode == 'span':
                        if self.span_decoder == 'crf':
                            num_tags = 2
                            self.fc_out = nn.Linear(model_dim, num_tags)
                            self.crf = CRF(num_tags=num_tags, batch_first=True)
                            with torch.no_grad():
                                self.crf.transitions.data.zero_()
                                self.crf.transitions.data[0, 0] = 1.0
                                self.crf.transitions.data[1, 1] = 1.0
                                self.crf.transitions.data[0, 1] = -1.0
                                self.crf.transitions.data[1, 0] = -1.0
                        else:
                            self.fc_out = nn.Linear(model_dim, 1)
                    else:
                        raise ValueError("mode must be 'pooling' or 'span'")

                def _generate_positional_encoding(self, max_len, model_dim):
                    position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
                    div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-np.log(10000.0) / model_dim))
                    pe = torch.zeros(max_len, model_dim)
                    pe[:, 0::2] = torch.sin(position * div_term)
                    pe[:, 1::2] = torch.cos(position * div_term)
                    return pe

                def forward(self, x, src_key_padding_mask=None):
                    x = self.input_fc(x)
                    x = self.dropout(x)
                    x = x + self.positional_encoding_buffer[:, :x.size(1), :].to(x.device)
                    attention_weights_list = []
                    for layer in self.layers:
                        x, attn_weights = layer(x, src_key_padding_mask=src_key_padding_mask)
                        attention_weights_list.append(attn_weights)

                    if self.mode == 'pooling':
                        per_timestep_logits = self.fc_out(x).squeeze(-1)

                        if self.pooling_type == 'mean':
                            pooled_output = per_timestep_logits.mean(dim=1)

                        elif self.pooling_type == 'max':
                            pooled_output = per_timestep_logits.max(dim=1)[0]

                        elif self.pooling_type == 'cls':
                            pooled_output = per_timestep_logits[:, 0]

                        elif self.pooling_type == 'attention':
                            attention_weights = attention_weights_list[-1].mean(dim=1)
                            cls_attention_weights = attention_weights[:, :, 0]
                            cls_attention_weights = F.softmax(cls_attention_weights, dim=-1)
                            weighted_attention = torch.bmm(
                                cls_attention_weights.unsqueeze(1),
                                per_timestep_logits.unsqueeze(-1)
                            ).squeeze(-1)
                            pooled_output = weighted_attention.mean(dim=1)

                        elif self.pooling_type == 'any':
                            if self.training:
                                # During training, same as max pooling
                                pooled_output = per_timestep_logits.max(dim=1)[0]
                            else:
                                # During inference, if any >=0.5 then 1
                                any_pred = (torch.sigmoid(per_timestep_logits) >= 0.5).any(dim=1).float()
                                pooled_output = any_pred.unsqueeze(-1)

                        elif self.pooling_type == 'count':
                            if self.training:
                                # During training, same as mean pooling
                                pooled_output = per_timestep_logits.mean(dim=1)
                            else:
                                # During inference, if half or more tokens >=0.5 then 1
                                preds_per_token = (torch.sigmoid(per_timestep_logits) >= 0.5).float()
                                majority = (preds_per_token.sum(dim=1) > (per_timestep_logits.size(1) / 2)).float()
                                pooled_output = majority.unsqueeze(-1)
                        else:
                            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

                        return pooled_output, attention_weights_list

                    elif self.mode == 'span':
                        if self.span_decoder == 'crf':
                            per_timestep_logits = self.fc_out(x)
                        else:
                            per_timestep_logits = self.fc_out(x).squeeze(-1)
                        return per_timestep_logits, attention_weights_list

            def labels_to_spans(labels):
                spans = []
                start = None
                for idx, label in enumerate(labels):
                    if label == 1:
                        if start is None:
                            start = idx
                    else:
                        if start is not None:
                            spans.append((start, idx))
                            start = None
                if start is not None:
                    spans.append((start, len(labels)))
                return spans

            def compute_token_spans_f1(gold_spans, pred_spans):
                gold_set = set()
                for start, end in gold_spans:
                    gold_set.update(range(start, end))
                pred_set = set()
                for start, end in pred_spans:
                    pred_set.update(range(start, end))
                tp = len(gold_set & pred_set)
                pred_len = len(pred_set)
                gold_len = len(gold_set)
                return tp, pred_len, gold_len

            def compute_span_f1(tp, pred_len, gold_len):
                precision = tp / pred_len if pred_len > 0 else 0.0
                recall = tp / gold_len if gold_len > 0 else 0.0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                return precision, recall, f1

            def get_crf_layer(model):
                if isinstance(model, nn.DataParallel):
                    return model.module.crf
                else:
                    return model.crf

            def evaluate(model, dataloader):
                model.eval()
                total_loss = 0.0

                if mode == 'pooling':
                    all_labels_list = []
                    all_preds_list = []
                else:
                    total_tp = 0
                    total_pred_len = 0
                    total_gold_len = 0

                with torch.no_grad():
                    for batch in dataloader:
                        if mode == 'pooling':
                            features, labels, masks = batch
                            features = features.to(device, non_blocking=True)
                            labels = labels.to(device, non_blocking=True)
                            masks = masks.to(device, non_blocking=True)
                            outputs, _ = model(features, src_key_padding_mask=masks)

                            if pooling_type in ['any', 'count']:
                                # outputs might already be (batch_size,) or (batch_size,1)
                                # applying inference logic, but no gradient needed here
                                preds = outputs.clone().squeeze(-1)
                                loss = F.binary_cross_entropy(
                                    preds, labels.squeeze(-1)
                                )
                            else:
                                loss = F.binary_cross_entropy_with_logits(outputs, labels.squeeze(-1))
                                preds = (torch.sigmoid(outputs) >= 0.5).float().squeeze(-1)

                            total_loss += loss.item()
                            all_labels = labels.squeeze(-1).cpu().numpy()
                            all_preds = preds.cpu().numpy()
                            all_labels_list.extend(all_labels)
                            all_preds_list.extend(all_preds)

                        else:  # mode == 'span'
                            features, labels, masks, labels_mask = batch
                            features = features.to(device, non_blocking=True)
                            labels = labels.to(device, non_blocking=True)
                            masks = masks.to(device, non_blocking=True)
                            labels_mask = labels_mask.to(device, non_blocking=True)

                            if model.module.span_decoder if isinstance(model, nn.DataParallel) else model.span_decoder == 'crf':
                                per_timestep_logits, _ = model(features, src_key_padding_mask=masks)
                                crf_layer = get_crf_layer(model)
                                loss = -crf_layer(per_timestep_logits, labels, mask=labels_mask.byte())
                                loss = loss / labels_mask.sum()
                                total_loss += loss.item()
                                preds = crf_layer.decode(per_timestep_logits, mask=labels_mask.byte())
                                batch_size = len(preds)
                                for i in range(batch_size):
                                    pred_seq = preds[i]
                                    label_seq = labels[i][labels_mask[i] == 1].cpu().tolist()
                                    pred_spans = labels_to_spans(pred_seq)
                                    gold_spans = labels_to_spans(label_seq)
                                    tp, pred_len, gold_len = compute_token_spans_f1(gold_spans, pred_spans)
                                    total_tp += tp
                                    total_pred_len += pred_len
                                    total_gold_len += gold_len
                            else:
                                per_timestep_logits, _ = model(features, src_key_padding_mask=masks)
                                valid_positions = (labels_mask == 1)
                                valid_logits = per_timestep_logits[valid_positions]
                                valid_labels = labels[valid_positions]
                                valid_labels = valid_labels.float().clamp(min=0)
                                if valid_labels.numel() > 0:
                                    loss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels)
                                    total_loss += loss.item()

                                preds_sigmoid = torch.sigmoid(per_timestep_logits)
                                preds_binary = (preds_sigmoid >= 0.5).long()
                                batch_size = features.size(0)

                                for i in range(batch_size):
                                    label_seq = labels[i][labels_mask[i] == 1].cpu().tolist()
                                    pred_seq = preds_binary[i][labels_mask[i] == 1].cpu().tolist()
                                    filtered_pairs = [(l, p) for (l, p) in zip(label_seq, pred_seq) if l != -1]
                                    if len(filtered_pairs) == 0:
                                        continue
                                    label_seq_filtered, pred_seq_filtered = zip(*filtered_pairs)
                                    gold_spans = labels_to_spans(label_seq_filtered)
                                    pred_spans = labels_to_spans(pred_seq_filtered)
                                    tp, pred_len, gold_len = compute_token_spans_f1(gold_spans, pred_spans)
                                    total_tp += tp
                                    total_pred_len += pred_len
                                    total_gold_len += gold_len

                if mode == 'pooling':
                    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
                    precision = precision_score(all_labels_list, all_preds_list, zero_division=0)
                    recall = recall_score(all_labels_list, all_preds_list, zero_division=0)
                    f1 = f1_score(all_labels_list, all_preds_list, zero_division=0)
                    return avg_loss, f1, precision, recall
                else:
                    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
                    precision_span, recall_span, f1_score_span = compute_span_f1(
                        total_tp, total_pred_len, total_gold_len
                    )
                    return avg_loss, f1_score_span, precision_span, recall_span

            if clf_mode == 'transformer':
                study = optuna.create_study(direction='maximize')

                def objective(trial):
                    train_dataloader = None
                    val_dataloader = None
                    test_dataloader = None
                    transformer_model = None
                    optimizer = None
                    scheduler = None

                    try:
                        potential_num_heads = [4, 8, 16, 32]
                        model_dim = trial.suggest_categorical('model_dim', [256, 512, 1024])
                        valid_num_heads = [h for h in potential_num_heads if model_dim % h == 0]

                        if len(valid_num_heads) == 0:
                            logger.warning(
                                f"Trial {trial.number}: No valid heads for model_dim={model_dim}. Skipping"
                            )
                            return 0.0

                        num_heads = trial.suggest_categorical('num_heads', valid_num_heads)

                        if mode == 'pooling':
                            num_layers = trial.suggest_categorical('num_layers', [2, 4, 6, 8, 10, 12, 14, 16])
                            dropout_prob = trial.suggest_float('dropout_prob', 0.1, 0.5)
                            lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
                            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
                            num_workers = min(8, os.cpu_count())
                            train_dataloader = DataLoader(
                                train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=num_workers,
                                prefetch_factor=4,
                                persistent_workers=True,
                                drop_last=True
                            )
                            val_dataloader = DataLoader(
                                val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=num_workers,
                                prefetch_factor=4,
                                persistent_workers=True,
                                drop_last=False
                            )
                            test_dataloader = DataLoader(
                                test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=num_workers,
                                prefetch_factor=4,
                                persistent_workers=True,
                                drop_last=False
                            )

                            transformer_model = TransformerClassifier(
                                input_dim=new_input_dim,
                                model_dim=model_dim,
                                num_heads=num_heads,
                                num_layers=num_layers,
                                dropout_prob=dropout_prob,
                                pooling_type=pooling_type,
                                mode=mode,
                                span_decoder=span_decoder  # Not needed in pooling, but passed as an argument
                            ).to(device)

                            if num_available_gpus > 1:
                                transformer_model = nn.DataParallel(transformer_model)

                            optimizer = optim.Adam(
                                transformer_model.parameters(),
                                lr=lr,
                                weight_decay=weight_decay
                            )
                            early_stopping = EarlyStopping(patience=10, verbose=True, delta=0.01)
                            scaler = torch.cuda.amp.GradScaler()

                            for epoch in range(n_epochs):
                                transformer_model.train()
                                running_loss = 0.0
                                for features, labels, masks in train_dataloader:
                                    features = features.to(device, non_blocking=True)
                                    labels = labels.to(device, non_blocking=True)
                                    masks = masks.to(device, non_blocking=True)
                                    optimizer.zero_grad()
                                    with torch.cuda.amp.autocast():
                                        outputs, _ = transformer_model(
                                            features, src_key_padding_mask=masks
                                        )
                                        # ★★★ Modified part ★★★
                                        if pooling_type in ['any','count']:
                                            loss = F.binary_cross_entropy_with_logits(outputs, labels.squeeze(-1))
                                        else:
                                            loss = F.binary_cross_entropy_with_logits(outputs, labels)
                                        # ★★★ Modification ends here ★★★

                                    scaler.scale(loss).backward()
                                    scaler.step(optimizer)
                                    scaler.update()
                                    running_loss += loss.item()

                                avg_loss = running_loss / len(train_dataloader)
                                val_loss, f1, precision, recall = evaluate(transformer_model, val_dataloader)
                                logger.info(
                                    f"Trial {trial.number}, Epoch {epoch+1}, "
                                    f"train_loss: {avg_loss:.4f}, val_loss: {val_loss:.4f}, "
                                    f"val_f1: {f1:.4f}, val_precision: {precision:.4f}, val_recall: {recall:.4f}"
                                )

                                early_stopping(f1, current_epoch=epoch+1, model=transformer_model)
                                if early_stopping.early_stop:
                                    logger.info(
                                        f"Early stopping at Epoch {epoch+1} for Trial {trial.number}."
                                    )
                                    break

                            if early_stopping.best_model_state is not None:
                                if isinstance(transformer_model, nn.DataParallel):
                                    transformer_model.module.load_state_dict(
                                        early_stopping.best_model_state
                                    )
                                else:
                                    transformer_model.load_state_dict(
                                        early_stopping.best_model_state
                                    )
                                logger.info(
                                    f"Trial {trial.number}: best_epoch={early_stopping.best_epoch} model restored"
                                )

                            val_loss, f1, precision, recall = evaluate(transformer_model, val_dataloader)
                            test_loss, f1_test, precision_test, recall_test = evaluate(transformer_model, test_dataloader)

                            model_filename = (
                                f"model_trial_{trial.number}_epoch_{early_stopping.best_epoch}_f1_{f1:.4f}.pt"
                            )
                            model_path = os.path.join(models_dir, model_filename)
                            if isinstance(transformer_model, nn.DataParallel):
                                torch.save(transformer_model.module.state_dict(), model_path)
                            else:
                                torch.save(transformer_model.state_dict(), model_path)

                            trial.set_user_attr("model_path", model_path)
                            trial.set_user_attr("best_epoch", early_stopping.best_epoch)
                            trial.set_user_attr("val_precision", precision)
                            trial.set_user_attr("val_recall", recall)
                            trial.set_user_attr("val_loss", val_loss)
                            trial.set_user_attr("test_precision", precision_test)
                            trial.set_user_attr("test_recall", recall_test)
                            trial.set_user_attr("test_f1_score", f1_test)

                            trial_info_str = (
                                f"Trial {trial.number}: params={trial.params}, "
                                f"val_loss={val_loss:.4f}, val_f1={f1:.4f}, "
                                f"val_precision={precision:.4f}, val_recall={recall:.4f}, "
                                f"test_f1={f1_test:.4f}, test_precision={precision_test:.4f}, "
                                f"test_recall={recall_test:.4f}, best_epoch={early_stopping.best_epoch}, "
                                f"model_path={model_path}"
                            )
                            logger.info(trial_info_str)

                            return f1

                        elif mode == 'span':
                            num_layers = trial.suggest_categorical('num_layers', [4, 6, 8])
                            dropout_prob = trial.suggest_float('dropout_prob', 0.1, 0.5)
                            lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
                            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
                            num_workers = min(8, os.cpu_count())
                            train_dataloader = DataLoader(
                                train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True,
                                num_workers=num_workers,
                                prefetch_factor=4,
                                persistent_workers=True,
                                drop_last=False
                            )
                            val_dataloader = DataLoader(
                                val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=num_workers,
                                prefetch_factor=4,
                                persistent_workers=True,
                                drop_last=False
                            )
                            test_dataloader = DataLoader(
                                test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                pin_memory=True,
                                num_workers=num_workers,
                                prefetch_factor=4,
                                persistent_workers=True,
                                drop_last=False
                            )

                            transformer_model = TransformerClassifier(
                                input_dim=new_input_dim,
                                model_dim=model_dim,
                                num_heads=num_heads,
                                num_layers=num_layers,
                                dropout_prob=dropout_prob,
                                pooling_type=None,
                                max_length=max_length,
                                mode=mode,
                                span_decoder=span_decoder
                            )
                            if num_available_gpus > 1:
                                transformer_model = nn.DataParallel(transformer_model)
                            transformer_model = transformer_model.to(device)

                            optimizer = optim.AdamW(
                                transformer_model.parameters(),
                                lr=lr,
                                weight_decay=weight_decay
                            )
                            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                                optimizer,
                                mode='min',
                                factor=0.5,
                                patience=5
                            )
                            early_stopping = EarlyStopping(patience=10, verbose=True, delta=0.01)
                            scaler = torch.cuda.amp.GradScaler()

                            for epoch in range(n_epochs):
                                transformer_model.train()
                                running_loss = 0.0
                                for features, labels, masks, labels_mask in train_dataloader:
                                    features = features.to(device, non_blocking=True)
                                    labels = labels.to(device, non_blocking=True)
                                    masks = masks.to(device, non_blocking=True)
                                    labels_mask = labels_mask.to(device, non_blocking=True)
                                    optimizer.zero_grad()
                                    with torch.cuda.amp.autocast():
                                        per_timestep_logits, _ = transformer_model(
                                            features,
                                            src_key_padding_mask=masks
                                        )

                                        if span_decoder == 'crf':
                                            crf_layer = get_crf_layer(transformer_model)
                                            loss = -crf_layer(per_timestep_logits, labels, mask=labels_mask.byte())
                                            loss = loss / labels_mask.sum()
                                        else:
                                            valid_positions = (labels_mask == 1)
                                            valid_logits = per_timestep_logits[valid_positions]
                                            valid_labels = labels[valid_positions].float().clamp(min=0)
                                            if valid_labels.numel() == 0:
                                                continue
                                            loss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels)

                                    if torch.isnan(loss):
                                        logger.warning(f"Trial {trial.number}, Epoch {epoch+1}: loss is NaN, skipping")
                                        return 0.0
                                    scaler.scale(loss).backward()
                                    scaler.step(optimizer)
                                    scaler.update()
                                    running_loss += loss.item()

                                torch.cuda.empty_cache()
                                if len(train_dataloader) == 0:
                                    continue
                                avg_loss = running_loss / len(train_dataloader)
                                val_loss, f1_score_span, precision_span, recall_span = evaluate(
                                    transformer_model, val_dataloader
                                )
                                logger.info(
                                    f"Trial {trial.number}, Epoch {epoch+1}, "
                                    f"train_loss={avg_loss:.4f}, val_loss={val_loss:.4f}, "
                                    f"val_f1={f1_score_span:.4f}, val_precision={precision_span:.4f}, "
                                    f"val_recall={recall_span:.4f}"
                                )

                                scheduler.step(val_loss)
                                early_stopping(
                                    f1_score_span, current_epoch=epoch+1, model=transformer_model
                                )
                                if early_stopping.early_stop:
                                    logger.info(
                                        f"Early stopping at Epoch {epoch+1} for Trial {trial.number}."
                                    )
                                    break

                            if early_stopping.best_model_state is not None:
                                if isinstance(transformer_model, nn.DataParallel):
                                    transformer_model.module.load_state_dict(
                                        early_stopping.best_model_state
                                    )
                                else:
                                    transformer_model.load_state_dict(early_stopping.best_model_state)
                                logger.info(
                                    f"best_epoch={early_stopping.best_epoch} model restored"
                                )

                            val_metrics = evaluate(transformer_model, val_dataloader)
                            test_metrics = evaluate(transformer_model, test_dataloader)
                            if mode == 'pooling':
                                val_loss, f1, precision, recall = val_metrics
                                test_loss, f1_test, precision_test, recall_test = test_metrics
                            else:
                                val_loss, f1, precision, recall = val_metrics
                                test_loss, f1_test, precision_test, recall_test = test_metrics

                            model_filename = (
                                f"model_trial_{trial.number}_epoch_{early_stopping.best_epoch}_f1_{f1:.4f}.pt"
                            )
                            model_path = os.path.join(models_dir, model_filename)
                            if isinstance(transformer_model, nn.DataParallel):
                                torch.save(transformer_model.module.state_dict(), model_path)
                            else:
                                torch.save(transformer_model.state_dict(), model_path)

                            trial.set_user_attr("model_path", model_path)
                            trial.set_user_attr("best_epoch", early_stopping.best_epoch)
                            trial.set_user_attr("val_loss", val_loss)
                            trial.set_user_attr("val_f1", f1)
                            trial.set_user_attr("val_precision", precision)
                            trial.set_user_attr("val_recall", recall)
                            trial.set_user_attr("test_loss", test_loss)
                            trial.set_user_attr("test_f1_score", f1_test)
                            trial.set_user_attr("test_precision", precision_test)
                            trial.set_user_attr("test_recall", recall_test)

                            trial_info_str = (
                                f"Trial {trial.number}: params={trial.params}, "
                                f"val_loss={val_loss:.4f}, val_f1={f1:.4f}, "
                                f"val_precision={precision:.4f}, val_recall={recall:.4f}, "
                                f"test_f1={f1_test:.4f}, test_precision={precision_test:.4f}, "
                                f"test_recall={recall_test:.4f}, best_epoch={early_stopping.best_epoch}, "
                                f"model_path={model_path}"
                            )
                            logger.info(trial_info_str)

                            return f1

                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():
                            logger.warning(f"Trial {trial.number}: OOM error => skip (return 0.0)")
                            if 'train_dataloader' in locals():
                                del train_dataloader
                            if 'val_dataloader' in locals():
                                del val_dataloader
                            if 'test_dataloader' in locals():
                                del test_dataloader
                            if 'transformer_model' in locals():
                                del transformer_model
                            if 'optimizer' in locals():
                                del optimizer
                            if 'scheduler' in locals():
                                del scheduler
                            torch.cuda.empty_cache()
                            gc.collect()
                            return 0.0
                        else:
                            logger.error(f"Trial {trial.number}: Unexpected RuntimeError => skip (return 0.0)")
                            if 'train_dataloader' in locals():
                                del train_dataloader
                            if 'val_dataloader' in locals():
                                del val_dataloader
                            if 'test_dataloader' in locals():
                                del test_dataloader
                            if 'transformer_model' in locals():
                                del transformer_model
                            if 'optimizer' in locals():
                                del optimizer
                            if 'scheduler' in locals():
                                del scheduler
                            torch.cuda.empty_cache()
                            gc.collect()
                            return 0.0
                    except Exception as e:
                        logger.error(f"Trial {trial.number}: Unexpected error: {e} Skipping", exc_info=True)
                        if 'train_dataloader' in locals():
                            del train_dataloader
                        if 'val_dataloader' in locals():
                            del val_dataloader
                        if 'test_dataloader' in locals():
                            del test_dataloader
                        if 'transformer_model' in locals():
                            del transformer_model
                        if 'optimizer' in locals():
                            del optimizer
                        if 'scheduler' in locals():
                            del scheduler
                        torch.cuda.empty_cache()
                        gc.collect()
                        return 0.0
                    finally:
                        if 'train_dataloader' in locals():
                            del train_dataloader
                        if 'val_dataloader' in locals():
                            del val_dataloader
                        if 'test_dataloader' in locals():
                            del test_dataloader
                        if 'transformer_model' in locals():
                            del transformer_model
                        if 'optimizer' in locals():
                            del optimizer
                        if 'scheduler' in locals():
                            del scheduler
                        torch.cuda.empty_cache()
                        gc.collect()

                # ★ If top_models_json_path is specified and exists, skip Optuna and retrain only
                if top_models_json_path and os.path.exists(top_models_json_path):
                    logger.info(
                        f"Parameter set file {top_models_json_path} was specified. Optuna will be skipped."
                    )
                    with open(top_models_json_path, 'r') as f:
                        top_trials_info = json.load(f)

                    retrained_top_info = []

                    for rank, trial_info in enumerate(top_trials_info, 1):
                        params = trial_info['params']
                        model_dim = params.get('model_dim', 1024)
                        num_heads = params.get('num_heads', 4)
                        num_layers = params.get('num_layers', 4)
                        dropout_prob = params.get('dropout_prob', 0.3)
                        lr = params.get('lr', 1e-3)
                        weight_decay = params.get('weight_decay', 1e-4)

                        logger.info(f"----- Retraining top model: params={params} -----")
                        print(f"----- Retraining top model: params={params} -----")

                        train_dataloader = None
                        val_dataloader = None
                        test_dataloader = None
                        transformer_model = None
                        optimizer = None
                        scheduler = None
                        try:
                            if mode == 'pooling':
                                transformer_model = TransformerClassifier(
                                    input_dim=new_input_dim,
                                    model_dim=model_dim,
                                    num_heads=num_heads,
                                    num_layers=num_layers,
                                    dropout_prob=dropout_prob,
                                    pooling_type=pooling_type,
                                    mode=mode,
                                    span_decoder=span_decoder
                                ).to(device)
                                optimizer = optim.Adam(
                                    transformer_model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay
                                )
                                loss_fn = F.binary_cross_entropy_with_logits
                            else:
                                transformer_model = TransformerClassifier(
                                    input_dim=new_input_dim,
                                    model_dim=model_dim,
                                    num_heads=num_heads,
                                    num_layers=num_layers,
                                    dropout_prob=dropout_prob,
                                    pooling_type=None,
                                    max_length=max_length,
                                    mode=mode,
                                    span_decoder=span_decoder
                                ).to(device)
                                optimizer = optim.AdamW(
                                    transformer_model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay
                                )
                                loss_fn = F.binary_cross_entropy_with_logits if span_decoder == 'linear' else None

                            if num_available_gpus > 1:
                                transformer_model = nn.DataParallel(transformer_model)

                            if mode == 'pooling':
                                train_dataloader = DataLoader(
                                    train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=min(8, os.cpu_count()),
                                    prefetch_factor=4,
                                    persistent_workers=True,
                                    drop_last=True
                                )
                                val_dataloader = DataLoader(
                                    val_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=min(8, os.cpu_count()),
                                    prefetch_factor=4,
                                    persistent_workers=True,
                                    drop_last=False
                                )
                                test_dataloader = DataLoader(
                                    test_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=min(8, os.cpu_count()),
                                    prefetch_factor=4,
                                    persistent_workers=True,
                                    drop_last=False
                                )
                            else:
                                train_dataloader = DataLoader(
                                    train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    pin_memory=True,
                                    num_workers=min(8, os.cpu_count()),
                                    prefetch_factor=4,
                                    persistent_workers=True,
                                    drop_last=False
                                )
                                val_dataloader = DataLoader(
                                    val_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=min(8, os.cpu_count()),
                                    prefetch_factor=4,
                                    persistent_workers=True,
                                    drop_last=False
                                )
                                test_dataloader = DataLoader(
                                    test_dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    pin_memory=True,
                                    num_workers=min(8, os.cpu_count()),
                                    prefetch_factor=4,
                                    persistent_workers=True,
                                    drop_last=False
                                )

                            early_stopping = EarlyStopping(patience=10, verbose=True, delta=0.01)
                            scaler = torch.cuda.amp.GradScaler()

                            for epoch in range(n_epochs):
                                transformer_model.train()
                                running_loss = 0.0

                                if mode == 'pooling':
                                    for features, labels, masks in train_dataloader:
                                        features = features.to(device, non_blocking=True)
                                        labels = labels.to(device, non_blocking=True)
                                        masks = masks.to(device, non_blocking=True)
                                        optimizer.zero_grad()
                                        with torch.cuda.amp.autocast():
                                            outputs, _ = transformer_model(
                                                features,
                                                src_key_padding_mask=masks
                                            )
                                            # ★★★ Similarly modified in retraining part ★★★
                                            if pooling_type in ['any','count']:
                                                loss = F.binary_cross_entropy_with_logits(outputs, labels.squeeze(-1))
                                            else:
                                                loss = loss_fn(outputs, labels)
                                            # ★★★ Modification ends here ★★★
                                        scaler.scale(loss).backward()
                                        scaler.step(optimizer)
                                        scaler.update()
                                        running_loss += loss.item()

                                else:
                                    # span mode
                                    for features, labels, masks, labels_mask in train_dataloader:
                                        features = features.to(device, non_blocking=True)
                                        labels = labels.to(device, non_blocking=True)
                                        masks = masks.to(device, non_blocking=True)
                                        labels_mask = labels_mask.to(device, non_blocking=True)
                                        optimizer.zero_grad()
                                        with torch.cuda.amp.autocast():
                                            per_timestep_logits, _ = transformer_model(
                                                features,
                                                src_key_padding_mask=masks
                                            )

                                            if span_decoder == 'crf':
                                                crf_layer = get_crf_layer(transformer_model)
                                                loss = -crf_layer(per_timestep_logits, labels, mask=labels_mask.byte())
                                                loss = loss / labels_mask.sum()
                                            else:
                                                valid_positions = (labels_mask == 1)
                                                valid_logits = per_timestep_logits[valid_positions]
                                                valid_labels = labels[valid_positions].float()
                                                valid_labels = valid_labels.clamp(min=0)
                                                if valid_labels.numel() == 0:
                                                    continue
                                                loss = loss_fn(valid_logits, valid_labels)

                                        if torch.isnan(loss):
                                            logger.warning(f"Epoch {epoch+1}: loss is NaN, skipping")
                                            continue
                                        scaler.scale(loss).backward()
                                        scaler.step(optimizer)
                                        scaler.update()
                                        running_loss += loss.item()

                                avg_loss = running_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0.0
                                val_metrics = evaluate(transformer_model, val_dataloader)
                                if mode == 'pooling':
                                    val_loss, f1, precision, recall = val_metrics
                                    logger.info(
                                        f"Epoch {epoch+1}, train_loss: {avg_loss:.4f}, "
                                        f"val_loss: {val_loss:.4f}, val_f1: {f1:.4f}, "
                                        f"val_precision: {precision:.4f}, val_recall: {recall:.4f}"
                                    )
                                    early_stopping(f1, current_epoch=epoch+1, model=transformer_model)
                                else:
                                    val_loss, f1_span, precision_span, recall_span = val_metrics
                                    logger.info(
                                        f"Epoch {epoch+1}, train_loss: {avg_loss:.4f}, "
                                        f"val_loss: {val_loss:.4f}, val_f1: {f1_span:.4f}, "
                                        f"val_precision: {precision_span:.4f}, val_recall: {recall_span:.4f}"
                                    )
                                    early_stopping(f1_span, current_epoch=epoch+1, model=transformer_model)

                                if early_stopping.early_stop:
                                    logger.info(f"Early stopping at Epoch {epoch+1}.")
                                    break

                            if early_stopping.best_model_state is not None:
                                if isinstance(transformer_model, nn.DataParallel):
                                    transformer_model.module.load_state_dict(
                                        early_stopping.best_model_state
                                    )
                                else:
                                    transformer_model.load_state_dict(early_stopping.best_model_state)
                                logger.info(
                                    f"best_epoch={early_stopping.best_epoch} model restored"
                                )

                            val_metrics = evaluate(transformer_model, val_dataloader)
                            test_metrics = evaluate(transformer_model, test_dataloader)
                            if mode == 'pooling':
                                val_loss, f1, precision, recall = val_metrics
                                test_loss, f1_test, precision_test, recall_test = test_metrics
                            else:
                                val_loss, f1, precision, recall = val_metrics
                                test_loss, f1_test, precision_test, recall_test = test_metrics

                            model_filename = (
                                f"top_model_{dataset_name}_{save_name}_{feature_type}_"
                                f"model_dim_{model_dim}_layers_{num_layers}_heads_{num_heads}_rank_{rank}.pt"
                            )
                            model_path = os.path.join(models_dir, model_filename)
                            if isinstance(transformer_model, nn.DataParallel):
                                torch.save(transformer_model.module.state_dict(), model_path)
                            else:
                                torch.save(transformer_model.state_dict(), model_path)

                            logger.info(f"Saved top model to {model_path}.")
                            print(f"Saved top model to {model_path}.")

                            eval_metrics = {
                                'params': params,
                                'model_path': model_path,
                                'best_epoch': early_stopping.best_epoch,
                                'val_loss': val_loss,
                                'val_f1': f1,
                                'val_precision': precision,
                                'val_recall': recall,
                                'test_loss': test_loss,
                                'test_f1': f1_test,
                                'test_precision': precision_test,
                                'test_recall': recall_test
                            }

                            eval_metrics_filename = (
                                f"eval_metrics_model_dim_{model_dim}_layers_{num_layers}_heads_{num_heads}_rank_{rank}.json"
                            )
                            eval_metrics_path = os.path.join(models_dir, eval_metrics_filename)
                            with open(eval_metrics_path, 'w') as f:
                                json.dump(eval_metrics, f, indent=4)
                            logger.info(f"Saved evaluation metrics to {eval_metrics_path}.")
                            print(f"Saved evaluation metrics to {eval_metrics_path}.")

                            retrained_top_info.append(eval_metrics)

                        except RuntimeError as e:
                            if 'out of memory' in str(e).lower():
                                logger.warning("OOM error => skip (continue)")
                                if 'train_dataloader' in locals():
                                    del train_dataloader
                                if 'val_dataloader' in locals():
                                    del val_dataloader
                                if 'test_dataloader' in locals():
                                    del test_dataloader
                                if 'transformer_model' in locals():
                                    del transformer_model
                                if 'optimizer' in locals():
                                    del optimizer
                                if 'scheduler' in locals():
                                    del scheduler
                                torch.cuda.empty_cache()
                                gc.collect()
                                continue
                            else:
                                logger.error(f"Unexpected RuntimeError: {e}", exc_info=True)
                                if 'train_dataloader' in locals():
                                    del train_dataloader
                                if 'val_dataloader' in locals():
                                    del val_dataloader
                                if 'test_dataloader' in locals():
                                    del test_dataloader
                                if 'transformer_model' in locals():
                                    del transformer_model
                                if 'optimizer' in locals():
                                    del optimizer
                                if 'scheduler' in locals():
                                    del scheduler
                                torch.cuda.empty_cache()
                                gc.collect()
                                continue
                        except Exception as e:
                            logger.error(f"Unexpected error occurred during retraining: {e}", exc_info=True)
                            if 'train_dataloader' in locals():
                                del train_dataloader
                            if 'val_dataloader' in locals():
                                del val_dataloader
                            if 'test_dataloader' in locals():
                                del test_dataloader
                            if 'transformer_model' in locals():
                                del transformer_model
                            if 'optimizer' in locals():
                                del optimizer
                            if 'scheduler' in locals():
                                del scheduler
                            torch.cuda.empty_cache()
                            gc.collect()
                            continue
                        finally:
                            if 'train_dataloader' in locals():
                                del train_dataloader
                            if 'val_dataloader' in locals():
                                del val_dataloader
                            if 'test_dataloader' in locals():
                                del test_dataloader
                            if 'transformer_model' in locals():
                                del transformer_model
                            if 'optimizer' in locals():
                                del optimizer
                            if 'scheduler' in locals():
                                del scheduler
                            torch.cuda.empty_cache()
                            gc.collect()

                    retrained_top_file = os.path.join(
                        output_dir,
                        f"retrained_top_models_{dataset_name}_{save_name}_{feature_type}_{pooling_type}.json"
                        if mode == 'pooling'
                        else f"retrained_top_models_{dataset_name}_{save_name}_{feature_type}.json"
                    )
                    retrained_top_info_sorted = sorted(
                        retrained_top_info, key=lambda x: x['val_f1'], reverse=True
                    )
                    retrained_top_5 = retrained_top_info_sorted[:5]
                    with open(retrained_top_file, 'w') as f:
                        json.dump(retrained_top_5, f, indent=4)
                    logger.info(
                        f"Saved top 5 retrained models info to {retrained_top_file}."
                    )
                    print(f"Saved top 5 retrained models info to {retrained_top_file}.")

                else:
                    study.optimize(objective, n_trials=n_trials, n_jobs=1)

                    sorted_trials = sorted(
                        study.trials,
                        key=lambda t: t.value if t.value is not None else -np.inf,
                        reverse=True
                    )
                    top5_trials = sorted_trials[:5]

                    top_trials_info = []
                    for rank, trial in enumerate(top5_trials, 1):
                        trial_number = trial.number
                        params = trial.params
                        f1 = trial.value if trial.value is not None else 0.0
                        model_path = trial.user_attrs.get("model_path", "N/A")
                        best_epoch = trial.user_attrs.get("best_epoch", None)
                        val_loss = trial.user_attrs.get("val_loss", None)
                        test_f1 = trial.user_attrs.get("test_f1_score", "N/A")
                        val_precision = trial.user_attrs.get("val_precision", None)
                        val_recall = trial.user_attrs.get("val_recall", None)
                        test_precision = trial.user_attrs.get("test_precision", None)
                        test_recall = trial.user_attrs.get("test_recall", None)

                        trial_info_dict = {
                            'rank': rank,
                            'trial_number': trial_number,
                            'params': params,
                            'val_loss': val_loss,
                            'val_f1': f1,
                            'best_epoch': best_epoch,
                            'model_path': model_path,
                            'val_precision': val_precision,
                            'val_recall': val_recall,
                            'test_f1': test_f1,
                            'test_precision': test_precision,
                            'test_recall': test_recall
                        }
                        top_trials_info.append(trial_info_dict)

                        trial_info_str = (
                            f"Top {rank} Trial {trial_number}: params={params}, "
                            f"val_loss={val_loss}, val_f1={f1}, test_f1={test_f1}, "
                            f"best_epoch={best_epoch}, model_path={model_path}, "
                            f"val_precision={val_precision}, val_recall={val_recall}, "
                            f"test_precision={test_precision}, test_recall={test_recall}"
                        )
                        logger.info(trial_info_str)

                    top_trials_file = os.path.join(
                        output_dir,
                        f"top5_transformer_models_{dataset_name}_{save_name}_{feature_type}_{pooling_type}.json"
                        if mode == 'pooling'
                        else f"top5_transformer_models_{dataset_name}_{save_name}_{feature_type}.json"
                    )
                    with open(top_trials_file, 'w') as f:
                        json.dump(top_trials_info, f, indent=4)
                    logger.info(f"Saved top 5 trials to {top_trials_file}.")

            elif clf_mode == 'lookback_ratio_lr':
                pass  # Already processed above
            else:
                raise ValueError("clf_mode must be 'transformer' or 'lookback_ratio_lr'")

    except Exception as e:
        logger.error("An error occurred in main()", exc_info=True)
        logger.error(f"Program can continue: {e}")
    finally:
        logger.info("Process finished")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"An error occurred but the program will not be forced to stop: {e}")
