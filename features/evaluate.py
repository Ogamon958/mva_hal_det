import argparse
import logging
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset
from torchcrf import CRF
import re
import pickle
import traceback
import pandas as pd
import gc
import optuna
from optuna.trial import TrialState

def main():
    # Enclose everything in a try-except block
    try:
        # Set arguments
        parser = argparse.ArgumentParser(
            description="Evaluate Top5 Transformer Models with CRF and Logistic Regression (all-features preprocessed) + llama/qwen dimension support"
        )
        parser.add_argument(
            "--mode",
            type=str,
            required=True,
            choices=["pooling", "span"],
            help="Mode: 'pooling' or 'span'",
        )
        parser.add_argument(
            "--dataset_name",
            type=str,
            required=True,
            help="Dataset name",
        )
        parser.add_argument(
            "--feature_type",
            type=str,
            required=True,
            choices=["raw", "norm"],
            help="Feature type: 'raw' or 'norm'",
        )
        parser.add_argument(
            "--save_name",
            type=str,
            default="default_save_name",
            help="Save file name",
        )
        parser.add_argument(
            "--pooling_type",
            type=str,
            default="mean",
            help="Pooling type: 'any', 'count', 'mean', 'max', 'cls', 'attention' (pooling mode only)",
        )
        parser.add_argument(
            "--train_path",
            type=str,
            required=True,
            help="Pickle path of training data",
        )
        parser.add_argument(
            "--val_path",
            type=str,
            required=True,
            help="Pickle path of validation data",
        )
        parser.add_argument(
            "--test_path",
            type=str,
            required=True,
            help="Pickle path of test data",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=64,
            help="Batch size at evaluation time (default 64)",
        )
        parser.add_argument(
            "--features_to_use",
            type=str,
            default="key_avg,query_entropy,key_entropy,lookback_ratio",
            help="Specify the features to use, separated by commas",
        )
        parser.add_argument(
            "--layers_to_use",
            type=str,
            default="0-31",
            help="The layer number to use. Example: '0-31' or '0,1,2'",
        )
        parser.add_argument(
            "--heads_to_use",
            type=str,
            default="0-31",
            help="The head number to use. Example: '0-31' or '0,1,2'",
        )
        parser.add_argument(
            "--clf_mode",
            type=str,
            default="transformer",
            choices=["transformer", "lookback_ratio_lr"],
            help="Classification mode: 'transformer' (default) or 'lookback_ratio_lr'",
        )
        parser.add_argument(
            "--span_decoder",
            type=str,
            choices=["crf", "linear"],
            default="crf",
            help="Decoder used in span mode: 'crf' or 'linear'",
        )
        parser.add_argument(
            "--top_models_json_path",
            type=str,
            default=None,
            help="Path to the JSON file containing the top model information from Optuna. If specified, only retraining is performed.",
        )
        parser.add_argument(
            "--models_dir",
            type=str,
            default=None,
            help="Path to the directory to save logistic regression or Transformer models. If not specified, the default output directory is used.",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default=None,
            help="Path to the directory to save results. If not specified, the default output directory is used.",
        )
        parser.add_argument(
            "--n_trials",
            type=int,
            default=100,
            help="Number of trials in Optuna (default 100)",
        )
        parser.add_argument(
            "--n_epochs",
            type=int,
            default=50,
            help="Number of epochs for training the Transformer model (default 50)",
        )
        # ★ Added model_type (switch between llama / qwen)
        parser.add_argument(
            "--model_type",
            type=str,
            default="llama",
            choices=["llama", "qwen"],
            help="Type of model from which the features are obtained: 'llama' or 'qwen' (supports features of different dimensions)",
        )

        args = parser.parse_args()

        mode = args.mode
        dataset_name = args.dataset_name
        feature_type = args.feature_type
        save_name = args.save_name
        pooling_type = args.pooling_type if mode == "pooling" else None
        train_path = args.train_path
        val_path = args.val_path
        test_path = args.test_path
        batch_size = args.batch_size
        requested_features = [
            f.strip() for f in args.features_to_use.split(",") if f.strip()
        ]
        clf_mode = args.clf_mode
        layers_to_use_str = args.layers_to_use
        heads_to_use_str = args.heads_to_use
        span_decoder = args.span_decoder
        top_models_json_path = args.top_models_json_path
        n_trials = args.n_trials
        n_epochs = args.n_epochs
        model_type = args.model_type

        def parse_range(range_str):
            if "-" in range_str:
                start, end = range_str.split("-")
                return list(range(int(start), int(end) + 1))
            else:
                return [int(x) for x in range_str.split(",")]

        layers_to_use = parse_range(layers_to_use_str)
        heads_to_use = parse_range(heads_to_use_str)

        # Logging settings
        logger = logging.getLogger("transformer_evaluate_logger")
        logger.setLevel(logging.INFO)

        # Set output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = f"/home/code/data/training_file/{save_name}_{feature_type}"
        os.makedirs(output_dir, exist_ok=True)

        if args.models_dir:
            models_dir = args.models_dir
        else:
            models_dir = os.path.join(output_dir, "models")
        os.makedirs(models_dir, exist_ok=True)

        # Include pooling_type or span_decoder in the log file name
        if mode == "pooling":
            log_filename = (
                f"transformer_evaluate_{dataset_name}_{save_name}_{feature_type}_{pooling_type}_{span_decoder}.log"
            )
        else:
            log_filename = (
                f"transformer_evaluate_{dataset_name}_{save_name}_span_{span_decoder}.log"
            )
        log_file = os.path.join(output_dir, log_filename)

        if logger.hasHandlers():
            logger.handlers.clear()

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
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
        if mode == "pooling":
            logger.info(f"Pooling Type: {pooling_type}")
        logger.info(f"Train Path: {train_path}")
        logger.info(f"Validation Path: {val_path}")
        logger.info(f"Test Path: {test_path}")
        logger.info(f"Batch Size: {batch_size}")
        logger.info(f"Requested Features: {requested_features}")
        logger.info(f"Layers to use: {layers_to_use}")
        logger.info(f"Heads to use: {heads_to_use}")
        logger.info(f"Classification Mode: {clf_mode}")
        logger.info(f"Span Decoder: {span_decoder}")
        logger.info(f"Top Models JSON Path: {top_models_json_path}")
        logger.info(f"Models Directory: {models_dir}")
        logger.info(f"Output Directory: {output_dir}")
        logger.info(f"Optuna Trials: {n_trials}")
        logger.info(f"Transformer Epochs: {n_epochs}")
        logger.info(f"Model Type (llama/qwen): {model_type}")
        logger.info("---------------------------")

        def set_seed(seed=42):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        set_seed()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Device used: {device}")
        num_gpus = torch.cuda.device_count()
        logger.info(f"Number of GPUs: {num_gpus}")

        # Cache file path
        preprocessed_dir = f"/home/code/data/cashe_file/features/{model_type}"
        os.makedirs(preprocessed_dir, exist_ok=True)
        preprocessed_file = f"preprocessed_data_{model_type}_{dataset_name}_{feature_type}_all_features.pt"
        preprocessed_path = os.path.join(preprocessed_dir, preprocessed_file)

        if not os.path.exists(preprocessed_path):
            logger.error(
                f"{preprocessed_path} does not exist. Please check if the preprocessed data has been created."
            )
            raise FileNotFoundError(f"{preprocessed_path} not found.")

        preprocessed_data = torch.load(preprocessed_path)

        # Retrieve Dataset based on mode
        if mode == "pooling":
            train_dataset = preprocessed_data["train_dataset"]
            val_dataset = preprocessed_data["val_dataset"]
            test_dataset = preprocessed_data["test_dataset"]
            input_dim = preprocessed_data["input_dim"]
            train_all_texts = preprocessed_data.get("train_all_texts", [""] * len(train_dataset))
            val_all_texts = preprocessed_data.get("val_all_texts", [""] * len(val_dataset))
            test_all_texts = preprocessed_data.get("test_all_texts", [""] * len(test_dataset))
            max_length = None
        elif mode == "span":
            train_dataset = preprocessed_data["train_dataset"]
            val_dataset = preprocessed_data["val_dataset"]
            test_dataset = preprocessed_data["test_dataset"]
            input_dim = preprocessed_data["input_dim"]
            max_length = preprocessed_data["max_length"]
            train_all_texts = preprocessed_data.get("train_all_texts", [""] * len(train_dataset))
            val_all_texts = preprocessed_data.get("val_all_texts", [""] * len(val_dataset))
            test_all_texts = preprocessed_data.get("test_all_texts", [""] * len(test_dataset))
        else:
            raise ValueError("mode must be either 'pooling' or 'span'.")

        logger.info("Finished loading preprocessed data")

        # All possible features to use
        all_possible_features = [
            "key_avg",
            "query_entropy",
            "key_entropy",
            "lookback_ratio",
        ]
        chosen_feature_names = [f"{feature_type}_{feat}" for feat in all_possible_features]

        # ★ Convert to 32x32 or 28x28 depending on model_type
        def subset_features_dataset(dataset, mode, model_type):
            all_tensors = dataset.tensors
            features = all_tensors[0]

            selected_features_list = []
            # For each feature, extract the specified ones and concatenate
            # llama: 32*32=1024, qwen: 28*28=784
            for i, feat_name in enumerate(chosen_feature_names):
                base_name = feat_name.replace(f"{feature_type}_", "")
                if base_name in requested_features:
                    if model_type == "llama":
                        # Handle 1024
                        start = i * 1024
                        end = (i + 1) * 1024
                        feature_chunk = features[:, :, start:end]
                        # shape: (N, seq_len, 32*32)
                        feature_chunk = feature_chunk.view(
                            feature_chunk.size(0), feature_chunk.size(1), 32, 32
                        )
                        # layer restriction
                        feature_chunk = feature_chunk[:, :, layers_to_use, :]
                        # head restriction
                        feature_chunk = feature_chunk[:, :, :, heads_to_use]
                        # reshape to (N, seq_len, ?)
                        feature_chunk = feature_chunk.view(
                            feature_chunk.size(0), feature_chunk.size(1), -1
                        )
                        selected_features_list.append(feature_chunk)

                    elif model_type == "qwen":
                        # Handle 784 (28*28)
                        start = i * 784
                        end = (i + 1) * 784
                        feature_chunk = features[:, :, start:end]
                        # shape: (N, seq_len, 28*28)
                        feature_chunk = feature_chunk.view(
                            feature_chunk.size(0), feature_chunk.size(1), 28, 28
                        )
                        # layer restriction
                        feature_chunk = feature_chunk[:, :, layers_to_use, :]
                        # head restriction
                        feature_chunk = feature_chunk[:, :, :, heads_to_use]
                        # reshape
                        feature_chunk = feature_chunk.view(
                            feature_chunk.size(0), feature_chunk.size(1), -1
                        )
                        selected_features_list.append(feature_chunk)

            # Concatenate
            if selected_features_list:
                new_features = torch.cat(selected_features_list, dim=-1)
            else:
                # If nothing was specified, keep everything concatenated
                new_features = features

            if mode == "pooling":
                labels = all_tensors[1]
                masks = all_tensors[2]
                return TensorDataset(new_features, labels, masks)
            else:
                labels = all_tensors[1]
                masks = all_tensors[2]
                labels_mask = all_tensors[3]
                return TensorDataset(new_features, labels, masks, labels_mask)

        # Apply the conversion
        train_dataset = subset_features_dataset(train_dataset, mode, model_type)
        val_dataset = subset_features_dataset(val_dataset, mode, model_type)
        test_dataset = subset_features_dataset(test_dataset, mode, model_type)

        # Number of features to use × (layers_to_use × heads_to_use)
        new_input_dim = len(requested_features) * (len(layers_to_use) * len(heads_to_use))

        def create_dataloader(dataset, batch_size, shuffle=False):
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=True,
                num_workers=min(8, os.cpu_count()),
                prefetch_factor=4,
                persistent_workers=True,
                drop_last=False,
            )

        class CustomTransformerEncoderLayer(nn.Module):
            def __init__(
                self, d_model, nhead, dim_feedforward=2048, dropout=0.1
            ):
                super(CustomTransformerEncoderLayer, self).__init__()
                self.self_attn = nn.MultiheadAttention(
                    d_model, nhead, dropout=dropout, batch_first=True
                )
                self.linear1 = nn.Linear(d_model, dim_feedforward)
                self.dropout = nn.Dropout(dropout)
                self.linear2 = nn.Linear(dim_feedforward, d_model)
                self.norm1 = nn.LayerNorm(d_model)
                self.norm2 = nn.LayerNorm(d_model)
                self.dropout1 = nn.Dropout(dropout)
                self.dropout2 = nn.Dropout(dropout)

            def forward(
                self, src, src_mask=None, src_key_padding_mask=None
            ):
                src2, attn_weights = self.self_attn(
                    src,
                    src,
                    src,
                    attn_mask=src_mask,
                    key_padding_mask=src_key_padding_mask,
                    need_weights=True,
                    average_attn_weights=False,
                )
                src = src + self.dropout1(src2)
                src = self.norm1(src)
                src2 = self.linear2(
                    self.dropout(F.gelu(self.linear1(src)))
                )
                src = src + self.dropout2(src2)
                src = self.norm2(src)
                return src, attn_weights

        class TransformerClassifier(nn.Module):
            def __init__(
                self,
                input_dim,
                model_dim=1024,
                num_heads=4,
                num_layers=4,
                max_length=300,
                dropout_prob=0.3,
                pooling_type=None,
                mode="pooling",
                span_decoder="crf"
            ):
                super(TransformerClassifier, self).__init__()
                self.input_fc = nn.Linear(input_dim, model_dim)
                self.mode = mode
                self.pooling_type = pooling_type
                self.dropout = nn.Dropout(dropout_prob)
                self.positional_encoding = self._generate_positional_encoding(
                    max_length + (1 if pooling_type else 0), model_dim
                )
                self.register_buffer(
                    "positional_encoding_buffer", self.positional_encoding.unsqueeze(0)
                )
                self.layers = nn.ModuleList(
                    [
                        CustomTransformerEncoderLayer(
                            d_model=model_dim,
                            nhead=num_heads,
                            dropout=dropout_prob,
                        )
                        for _ in range(num_layers)
                    ]
                )
                self.span_decoder = span_decoder

                if mode == "pooling":
                    self.fc_out = nn.Linear(model_dim, 1)
                elif mode == "span":
                    if span_decoder == "crf":
                        num_tags = 2
                        self.fc_out = nn.Linear(model_dim, num_tags)
                        self.crf = CRF(num_tags=num_tags, batch_first=True)
                        # Initialize CRF
                        with torch.no_grad():
                            self.crf.transitions.data.zero_()
                            self.crf.transitions.data[0, 0] = 1.0
                            self.crf.transitions.data[1, 1] = 1.0
                            self.crf.transitions.data[0, 1] = -1.0
                            self.crf.transitions.data[1, 0] = -1.0
                    elif span_decoder == "linear":
                        self.fc_out = nn.Linear(model_dim, 1)
                    else:
                        raise ValueError("span_decoder must be either 'crf' or 'linear'.")
                else:
                    raise ValueError("mode must be either 'pooling' or 'span'.")

            def _generate_positional_encoding(self, max_len, model_dim):
                position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, model_dim, 2).float()
                    * (-np.log(10000.0) / model_dim)
                )
                pe = torch.zeros(max_len, model_dim)
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                return pe

            def forward(self, x, src_key_padding_mask=None):
                x = self.input_fc(x)
                x = self.dropout(x)
                x = x + self.positional_encoding_buffer[:, : x.size(1), :].to(x.device)
                attention_weights_list = []
                for layer in self.layers:
                    x, attn_weights = layer(x, src_key_padding_mask=src_key_padding_mask)
                    attention_weights_list.append(attn_weights)

                if self.mode == "pooling":
                    per_timestep_logits = self.fc_out(x).squeeze(-1)
                    if self.pooling_type in [
                        "mean",
                        "max",
                        "cls",
                        "attention",
                        "any",
                        "count",
                    ]:
                        if self.pooling_type == "mean":
                            pooled_output = per_timestep_logits.mean(dim=1)
                        elif self.pooling_type == "max":
                            pooled_output = per_timestep_logits.max(dim=1)[0]
                        elif self.pooling_type == "cls":
                            pooled_output = per_timestep_logits[:, 0]
                        elif self.pooling_type == "attention":
                            attention_weights = attention_weights_list[-1].mean(dim=1)
                            cls_attention_weights = attention_weights[:, :, 0]
                            cls_attention_weights = F.softmax(cls_attention_weights, dim=-1)
                            weighted_attention = torch.bmm(
                                cls_attention_weights.unsqueeze(1),
                                per_timestep_logits.unsqueeze(-1),
                            ).squeeze(-1)
                            pooled_output = weighted_attention.mean(dim=1)
                        elif self.pooling_type == "any":
                            if self.training:
                                pooled_output = per_timestep_logits.max(dim=1)[0]
                            else:
                                any_pred = (torch.sigmoid(per_timestep_logits) >= 0.5).any(dim=1).float()
                                pooled_output = any_pred.unsqueeze(-1)
                        elif self.pooling_type == "count":
                            if self.training:
                                pooled_output = per_timestep_logits.mean(dim=1)
                            else:
                                preds_per_token = (torch.sigmoid(per_timestep_logits) >= 0.5).float()
                                majority = (preds_per_token.sum(dim=1) > (per_timestep_logits.size(1) / 2)).float()
                                pooled_output = majority.unsqueeze(-1)
                        else:
                            raise ValueError(
                                f"Unknown pooling type: {self.pooling_type}"
                            )
                        return pooled_output, attention_weights_list

                    else:
                        raise ValueError(
                            f"Unknown pooling type: {self.pooling_type}"
                        )

                elif self.mode == "span":
                    if self.span_decoder == "crf":
                        per_timestep_logits = self.fc_out(x)  # shape: (B, L, 2)
                        return per_timestep_logits, attention_weights_list
                    elif self.span_decoder == "linear":
                        per_timestep_logits = self.fc_out(x).squeeze(-1)  # shape: (B, L)
                        return per_timestep_logits, attention_weights_list
                    else:
                        raise ValueError("span_decoder must be either 'crf' or 'linear'.")
                else:
                    raise ValueError("mode must be either 'pooling' or 'span'.")

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
            f1 = (
                (2 * precision * recall) / (precision + recall)
                if precision + recall > 0
                else 0.0
            )
            return precision, recall, f1

        def get_crf_layer(model):
            if hasattr(model, "module"):
                return model.module.crf
            else:
                return model.crf

        def get_span_decoder(model):
            if hasattr(model, "module"):
                return model.module.span_decoder
            else:
                return model.span_decoder

        def evaluate(
            model,
            dataloader,
            mode,
            span_threshold=0.5,
            return_outputs=False,
            clf_mode="transformer",
            window_size=1,
        ):
            model.eval()
            total_loss = 0.0

            per_sample_preds = []
            per_sample_labels = []
            per_sample_logits = []

            total_tp = 0
            total_pred_spans = 0
            total_gold_spans = 0

            all_true_labels = []
            all_pred_labels = []

            pooling_type_eval = None
            if hasattr(model, "pooling_type"):
                pooling_type_eval = model.pooling_type

            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if clf_mode == "lookback_ratio_lr" and mode == "span":
                        pass
                    else:
                        if mode == "pooling":
                            features, labels, masks = batch
                            features = features.to(device, non_blocking=True)
                            labels = labels.to(device, non_blocking=True)
                            masks = masks.to(device, non_blocking=True)

                            outputs, _ = model(features, src_key_padding_mask=masks)

                            if pooling_type_eval in ["any", "count"]:
                                loss = F.binary_cross_entropy(outputs, labels.squeeze(-1))
                                total_loss += loss.item()
                                preds = outputs.detach().cpu().numpy()
                            else:
                                loss = F.binary_cross_entropy_with_logits(
                                    outputs, labels.squeeze(-1)
                                )
                                total_loss += loss.item()
                                preds = (
                                    torch.sigmoid(outputs) >= span_threshold
                                ).float().detach().cpu().numpy()

                            logits = outputs.detach().cpu().numpy()
                            labels_np = labels.detach().cpu().numpy()

                            if return_outputs:
                                per_sample_preds.extend(preds.tolist())
                                per_sample_labels.extend(labels_np.tolist())
                                per_sample_logits.extend(logits.tolist())

                            all_true_labels.extend(labels_np.flatten().tolist())
                            all_pred_labels.extend(preds.flatten().tolist())

                        elif mode == "span":
                            features, labels, masks, labels_mask = batch
                            features = features.to(device, non_blocking=True)
                            labels = labels.to(device, non_blocking=True)
                            masks = masks.to(device, non_blocking=True)
                            labels_mask = labels_mask.to(device, non_blocking=True)

                            per_timestep_logits, _ = model(features, src_key_padding_mask=masks)

                            if clf_mode == "transformer":
                                if get_span_decoder(model) == "crf":
                                    crf_layer = get_crf_layer(model)
                                    loss = -crf_layer(
                                        per_timestep_logits, labels, mask=labels_mask.bool()
                                    )
                                    loss = loss / labels_mask.sum()
                                    total_loss += loss.item()

                                    preds = crf_layer.decode(
                                        per_timestep_logits, mask=labels_mask.bool()
                                    )
                                    batch_size = len(preds)
                                    for i in range(batch_size):
                                        pred_seq = preds[i]
                                        label_seq = labels[i][labels_mask[i] == 1].cpu().tolist()
                                        pred_spans = labels_to_spans(pred_seq)
                                        gold_spans = labels_to_spans(label_seq)

                                        tp, pred_len, gold_len = compute_token_spans_f1(
                                            gold_spans, pred_spans
                                        )
                                        total_tp += tp
                                        total_pred_spans += pred_len
                                        total_gold_spans += gold_len

                                        if return_outputs:
                                            per_sample_preds.append(pred_seq)
                                            per_sample_labels.append(label_seq)
                                            per_sample_logits.append(
                                                per_timestep_logits[i]
                                                .detach()
                                                .cpu()
                                                .numpy()
                                                .tolist()
                                            )

                                        all_true_labels.extend(label_seq)
                                        all_pred_labels.extend(pred_seq)

                                elif get_span_decoder(model) == "linear":
                                    valid_positions = (labels_mask == 1)
                                    valid_logits = per_timestep_logits[valid_positions]
                                    valid_labels = labels[valid_positions].float()
                                    valid_labels = valid_labels.clamp(min=0)

                                    if valid_labels.numel() > 0:
                                        loss = F.binary_cross_entropy_with_logits(valid_logits, valid_labels)
                                        total_loss += loss.item()

                                    preds = (torch.sigmoid(per_timestep_logits) >= 0.5).long()

                                    batch_size = features.size(0)
                                    for i in range(batch_size):
                                        label_seq = labels[i][labels_mask[i] == 1].cpu().tolist()
                                        pred_seq = preds[i][labels_mask[i] == 1].cpu().tolist()

                                        filtered_pairs = [(l, p) for (l, p) in zip(label_seq, pred_seq) if l != -1]
                                        if len(filtered_pairs) == 0:
                                            continue

                                        label_seq_filtered, pred_seq_filtered = zip(*filtered_pairs)

                                        gold_spans = labels_to_spans(label_seq_filtered)
                                        pred_spans = labels_to_spans(pred_seq_filtered)
                                        tp, pred_len, gold_len = compute_token_spans_f1(gold_spans, pred_spans)
                                        total_tp += tp
                                        total_pred_spans += pred_len
                                        total_gold_spans += gold_len

                                        if return_outputs:
                                            per_sample_preds.append(pred_seq_filtered)
                                            per_sample_labels.append(label_seq_filtered)
                                            per_sample_logits.append(
                                                per_timestep_logits[i].detach().cpu().numpy().tolist()
                                            )

                                        all_true_labels.extend(label_seq_filtered)
                                        all_pred_labels.extend(pred_seq_filtered)

            if mode == "pooling":
                avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
                f1_val = f1_score(all_true_labels, all_pred_labels, zero_division=0)
                accuracy_val = accuracy_score(all_true_labels, all_pred_labels)
                conf_matrix_val = confusion_matrix(all_true_labels, all_pred_labels)
                class_report_val = classification_report(
                    all_true_labels, all_pred_labels, zero_division=0
                )
                if return_outputs:
                    return (
                        avg_loss,
                        f1_val,
                        accuracy_val,
                        conf_matrix_val,
                        class_report_val,
                        per_sample_preds,
                        per_sample_labels,
                        per_sample_logits,
                        all_true_labels,
                        all_pred_labels,
                    )
                else:
                    return avg_loss, f1_val, accuracy_val, conf_matrix_val, class_report_val

            elif mode == "span":
                if clf_mode == "transformer":
                    if get_span_decoder(model) == "crf":
                        precision_span, recall_span, f1_span = compute_span_f1(
                            total_tp, total_pred_spans, total_gold_spans
                        )
                        conf_matrix_val = confusion_matrix(all_true_labels, all_pred_labels)
                        class_report_val = classification_report(
                            all_true_labels, all_pred_labels, zero_division=0
                        )
                        if return_outputs:
                            return (
                                f1_span,
                                precision_span,
                                recall_span,
                                conf_matrix_val,
                                class_report_val,
                                per_sample_preds,
                                per_sample_labels,
                                per_sample_logits,
                            )
                        else:
                            return f1_span, precision_span, recall_span, conf_matrix_val, class_report_val
                    elif get_span_decoder(model) == "linear":
                        precision_span, recall_span, f1_span = compute_span_f1(
                            total_tp, total_pred_spans, total_gold_spans
                        )
                        conf_matrix_val = confusion_matrix(all_true_labels, all_pred_labels)
                        class_report_val = classification_report(
                            all_true_labels, all_pred_labels, zero_division=0
                        )
                        if return_outputs:
                            return (
                                f1_span,
                                precision_span,
                                recall_span,
                                conf_matrix_val,
                                class_report_val,
                                per_sample_preds,
                                per_sample_labels,
                                per_sample_logits,
                            )
                        else:
                            return f1_span, precision_span, recall_span, conf_matrix_val, class_report_val
                elif clf_mode == "lookback_ratio_lr":
                    return None

        def sliding_window_evaluate(
            features, labels, labels_mask, window_size=1, device="cpu"
        ):
            N, seq_len, dim = features.shape
            windows = features.unfold(1, window_size, 1)
            X = windows.contiguous().view(
                N, -1, window_size * dim
            )

            if window_size == 1:
                y_full = labels.view(N, seq_len)
                mask_full = labels_mask.view(N, seq_len).bool()
                valid_indices = mask_full & (y_full != -1)
                valid_indices_flat = valid_indices.reshape(-1)
                y_filtered = y_full.reshape(-1)[valid_indices_flat]
                X_filtered = X.reshape(-1, window_size * dim)[valid_indices_flat]
            else:
                y_full = labels[:, window_size - 1 :]
                mask_full = labels_mask[:, window_size - 1 :].bool()
                valid_indices = mask_full & (y_full != -1)
                valid_indices_flat = valid_indices.reshape(-1)
                y_filtered = y_full.reshape(-1)[valid_indices_flat]
                X_flat = X.reshape(-1, window_size * dim)
                X_filtered = X_flat[valid_indices_flat]

            if torch.any((y_filtered != 0) & (y_filtered != 1)):
                raise ValueError("Values other than 0 or 1 are included in the labels.")

            return X_filtered.cpu().numpy(), y_filtered.cpu().numpy()

        # =====================================================================
        # Execution logic starts here
        # =====================================================================

        if clf_mode == "transformer":
            # Flow to load top models for the transformer and evaluate
            top_trials_file = os.path.join(
                output_dir,
                f"top5_transformer_models_{dataset_name}_{save_name}_{feature_type}_{pooling_type}.json"
                if mode == "pooling"
                else f"top5_transformer_models_{dataset_name}_{save_name}_{feature_type}.json",
            )
            retrained_trials_file = os.path.join(
                output_dir,
                f"retrained_top_models_{dataset_name}_{save_name}_{feature_type}_{pooling_type}.json"
                if mode == "pooling"
                else f"retrained_top_models_{dataset_name}_{save_name}_{feature_type}.json",
            )

            if not os.path.exists(top_trials_file):
                if os.path.exists(retrained_trials_file):
                    top_trials_file = retrained_trials_file
                    logger.warning(f"Top5 trials file not found. Using retrained file: {retrained_trials_file}.")
                    print(f"Top5 trials file not found. Using retrained file: {retrained_trials_file}.")
                else:
                    logger.error(f"Neither Top5 trials file {top_trials_file} nor retrained file {retrained_trials_file} found.")
                    print(f"Neither Top5 trials file {top_trials_file} nor {retrained_trials_file} found. Please run training script first.")
                    raise FileNotFoundError(f"Neither {top_trials_file} nor {retrained_trials_file} found.")

            with open(top_trials_file, "r") as f:
                top5_trials_info = json.load(f)

            test_dataloader = create_dataloader(
                test_dataset, batch_size=batch_size, shuffle=False
            )
            val_dataloader = create_dataloader(
                val_dataset, batch_size=batch_size, shuffle=False
            )

            # Evaluation routine
            if mode == "pooling":
                for trial_info in top5_trials_info:
                    trial_number = trial_info.get("trial_number", "N/A")
                    params = trial_info.get("params", {})
                    model_path = trial_info.get("model_path", "N/A")

                    if trial_number == "N/A":
                        rank_match = re.search(r"_rank_(\d+)", model_path)
                        if rank_match:
                            identifier = f"rank_{rank_match.group(1)}"
                        else:
                            identifier = "N_A"
                    else:
                        identifier = f"trial_{trial_number}"
                    identifier = re.sub(r"[^\w\-]", "_", identifier)

                    if model_path == "N/A" or not os.path.exists(model_path):
                        print(f"Trial {trial_number} has no saved model. Skipping evaluation.")
                        logger.info(f"Trial {trial_number} has no saved model. Skipping evaluation.")
                        continue

                    model_dim = params.get("model_dim", 1024)
                    model = TransformerClassifier(
                        input_dim=new_input_dim,
                        model_dim=model_dim,
                        num_heads=params.get("num_heads", 4),
                        num_layers=params.get("num_layers", 4),
                        dropout_prob=params.get("dropout_prob", 0.3),
                        pooling_type=pooling_type,
                        mode=mode,
                        span_decoder=span_decoder
                    )

                    if num_gpus > 1:
                        model = nn.DataParallel(model)
                    model.to(device)

                    state_dict = torch.load(model_path, map_location=device)
                    try:
                        if isinstance(model, nn.DataParallel):
                            model.module.load_state_dict(state_dict, strict=True)
                        else:
                            model.load_state_dict(state_dict, strict=True)
                    except RuntimeError as e:
                        logger.error(f"Error loading state_dict for Trial {identifier}: {e}")
                        print(f"Error loading state_dict for Trial {identifier}: {e}")
                        continue

                    logger.info(f"Loaded model from {model_path}")
                    print(f"Loaded model from {model_path}")

                    try:
                        avg_loss_test, f1_test, accuracy_test, conf_matrix_test, class_report_test = evaluate(
                            model, test_dataloader, mode, clf_mode=clf_mode
                        )
                        print(
                            f"Trial {identifier} - Test F1: {f1_test:.4f}, Accuracy: {accuracy_test:.4f}"
                        )
                        print("Confusion Matrix:")
                        print(conf_matrix_test)
                        print("Classification Report:")
                        print(class_report_test)

                        logger.info(
                            f"Trial {identifier} - Test F1: {f1_test:.4f}, Accuracy: {accuracy_test:.4f}"
                        )
                        logger.info(f"Confusion Matrix:\n{conf_matrix_test}")
                        logger.info(f"Classification Report:\n{class_report_test}")
                    except Exception as e:
                        logger.error(
                            f"Error during evaluation for Trial {identifier}: {e}",
                            exc_info=True,
                        )
                        print(f"Error during evaluation for Trial {identifier}: {e}")
                        continue

            else:  # mode == 'span'
                for trial_info in top5_trials_info:
                    trial_number = trial_info.get("trial_number", "N/A")
                    params = trial_info.get("params", {})
                    model_path = trial_info.get("model_path", "N/A")

                    if trial_number == "N/A":
                        rank_match = re.search(r"_rank_(\d+)", model_path)
                        if rank_match:
                            identifier = f"rank_{rank_match.group(1)}"
                        else:
                            identifier = "N_A"
                    else:
                        identifier = f"trial_{trial_number}"
                    identifier = re.sub(r"[^\w\-]", "_", identifier)

                    if model_path == "N/A" or not os.path.exists(model_path):
                        print(f"Trial {trial_number} has no saved model. Skipping evaluation.")
                        logger.info(f"Trial {trial_number} has no saved model. Skipping evaluation.")
                        continue

                    model_dim = params.get("model_dim", 1024)
                    model = TransformerClassifier(
                        input_dim=new_input_dim,
                        model_dim=model_dim,
                        num_heads=params.get("num_heads", 4),
                        num_layers=params.get("num_layers", 4),
                        dropout_prob=params.get("dropout_prob", 0.3),
                        pooling_type=None,
                        max_length=max_length,
                        mode=mode,
                        span_decoder=span_decoder
                    )

                    if num_gpus > 1:
                        model = nn.DataParallel(model)
                    model.to(device)

                    state_dict = torch.load(model_path, map_location=device)
                    try:
                        if isinstance(model, nn.DataParallel):
                            model.module.load_state_dict(state_dict, strict=True)
                        else:
                            model.load_state_dict(state_dict, strict=True)
                    except RuntimeError as e:
                        logger.error(f"Error loading state_dict for Trial {identifier}: {e}")
                        print(f"Error loading state_dict for Trial {identifier}: {e}")
                        continue

                    logger.info(f"Loaded model from {model_path}")
                    print(f"Loaded model from {model_path}")

                    try:
                        (
                            f1_span_test,
                            precision_span_test,
                            recall_span_test,
                            conf_matrix_test,
                            class_report_test,
                            per_sample_preds_test,
                            per_sample_labels_test,
                            per_sample_logits_test,
                        ) = evaluate(
                            model,
                            create_dataloader(test_dataset, batch_size, shuffle=False),
                            mode,
                            clf_mode=clf_mode,
                            return_outputs=True,
                        )
                        print(
                            f"Trial {identifier} - Test F1(span): {f1_span_test:.4f}, "
                            f"Precision: {precision_span_test:.4f}, Recall: {recall_span_test:.4f}"
                        )
                        print("Confusion Matrix:")
                        print(conf_matrix_test)
                        print("Classification Report:")
                        print(class_report_test)

                        logger.info(
                            f"Trial {identifier} - Test F1(span): {f1_span_test:.4f}, "
                            f"Precision: {precision_span_test:.4f}, Recall: {recall_span_test:.4f}"
                        )
                        logger.info(f"Confusion Matrix:\n{conf_matrix_test}")
                        logger.info(f"Classification Report:\n{class_report_test}")

                        if mode == "span":
                            if len(per_sample_preds_test) != len(test_all_texts):
                                logger.error(
                                    f"Mismatch in lengths: test_all_texts ({len(test_all_texts)}), "
                                    f"preds ({len(per_sample_preds_test)}), "
                                    f"labels ({len(per_sample_labels_test)}), "
                                    f"logits ({len(per_sample_logits_test)})"
                                )
                                print(
                                    f"Error: Mismatch in lengths: test_all_texts ({len(test_all_texts)}), "
                                    f"preds ({len(per_sample_preds_test)}), "
                                    f"labels ({len(per_sample_labels_test)}), "
                                    f"logits ({len(per_sample_logits_test)})"
                                )
                            else:
                                test_results = pd.DataFrame({
                                    "all_texts": test_all_texts,
                                    "labels": per_sample_labels_test,
                                    "predictions": per_sample_preds_test,
                                    "logits": per_sample_logits_test
                                })

                                test_results_file = os.path.join(
                                    output_dir, f"test_results_trial_{identifier}.csv"
                                )
                                test_results.to_csv(test_results_file, index=False)
                                logger.info(f"Test results saved to {test_results_file}")
                                print(f"Test results saved to {test_results_file}")

                    except Exception as e:
                        logger.error(
                            f"Error during evaluation for Trial {identifier}: {e}",
                            exc_info=True,
                        )
                        print(f"Error during evaluation for Trial {identifier}: {e}")
                        continue

        elif clf_mode == "lookback_ratio_lr":
            if mode != "span":
                logger.error("lookback_ratio_lr mode is supported only in 'span' mode.")
                print("lookback_ratio_lr mode is supported only in 'span' mode.")
                raise ValueError("lookback_ratio_lr mode is supported only in 'span' mode.")

            if not os.path.exists(models_dir):
                logger.error(
                    f"Model directory {models_dir} does not exist. Logistic Regression model may not be saved."
                )
                print(
                    f"Model directory {models_dir} does not exist. Logistic Regression model may not be saved."
                )
                raise FileNotFoundError(f"{models_dir} not found.")

            lr_model_files = []
            pattern = re.compile(
                r"lr_model_.+_lookback_ratio_span_window_(\d+)\.pkl$"
            )
            for root, dirs, files in os.walk(models_dir):
                for file in files:
                    match = pattern.match(file)
                    if match:
                        window_size = int(match.group(1))
                        lr_model_files.append((file, window_size))

            if not lr_model_files:
                logger.error(f"No Logistic Regression model files found in {models_dir}.")
                print(f"No Logistic Regression model files found in {models_dir}.")
                raise FileNotFoundError(f"No Logistic Regression model files found in {models_dir}.")

            test_dataloader = create_dataloader(
                test_dataset, batch_size=batch_size, shuffle=False
            )

            for lr_model_file, window_size in lr_model_files:
                trial_number = "N/A"
                if trial_number == "N/A":
                    rank_match = re.search(r"_rank_(\d+)", lr_model_file)
                    if rank_match:
                        identifier = f"rank_{rank_match.group(1)}"
                    else:
                        identifier = "N_A"
                else:
                    identifier = f"trial_{trial_number}"
                identifier = re.sub(r"[^\w\-]", "_", identifier)

                logger.info(
                    f"----- Logistic Regression Classification Evaluation: Trial {identifier}, Window Size={window_size} -----"
                )
                print(
                    f"----- Logistic Regression Classification Evaluation: Trial {identifier}, Window Size={window_size} -----"
                )

                lr_model_path = os.path.join(models_dir, lr_model_file)
                if not os.path.exists(lr_model_path):
                    logger.error(
                        f"Logistic Regression model {lr_model_path} does not exist. Skipping."
                    )
                    print(
                        f"Logistic Regression model {lr_model_path} does not exist. Skipping."
                    )
                    continue

                try:
                    with open(lr_model_path, "rb") as f:
                        lr_model = pickle.load(f)
                except Exception as e:
                    logger.error(f"Error loading Logistic Regression model {lr_model_path}: {e}")
                    print(f"Error loading Logistic Regression model {lr_model_path}: {e}")
                    continue

                logger.info(f"Loaded Logistic Regression model from {lr_model_path}")
                print(f"Loaded Logistic Regression model from {lr_model_path}")

                try:
                    X_test, y_test = sliding_window_evaluate(
                        test_dataset.tensors[0],
                        test_dataset.tensors[1],
                        test_dataset.tensors[3],
                        window_size=window_size,
                        device=device,
                    )
                except Exception as e:
                    logger.error(
                        f"Error during sliding_window_evaluate for Trial {identifier}, Window {window_size}: {e}",
                        exc_info=True,
                    )
                    print(
                        f"Error during sliding_window_evaluate for Trial {identifier}, Window {window_size}: {e}"
                    )
                    continue

                if X_test.size == 0:
                    logger.warning(
                        f"No valid data for window size {window_size}. Skipping."
                    )
                    print(
                        f"No valid data for window size {window_size}. Skipping."
                    )
                    continue

                try:
                    y_pred_proba = lr_model.predict_proba(X_test)[:, 1]
                    y_pred = (y_pred_proba >= 0.5).astype(int)
                except Exception as e:
                    logger.error(
                        f"Error during prediction for Trial {identifier}, Window {window_size}: {e}",
                        exc_info=True,
                    )
                    print(
                        f"Error during prediction for Trial {identifier}, Window {window_size}: {e}"
                    )
                    continue

                y_test_np = y_test

                try:
                    precision = precision_score(y_test_np, y_pred, zero_division=0)
                    recall = recall_score(y_test_np, y_pred, zero_division=0)
                    f1_score_val = f1_score(y_test_np, y_pred, zero_division=0)
                    conf_mat = confusion_matrix(y_test_np, y_pred)
                    try:
                        auroc = roc_auc_score(y_test_np, y_pred_proba)
                    except ValueError:
                        auroc = float("nan")

                    logger.info(
                        f"Logistic Regression evaluation result for Trial {identifier}, Window Size {window_size}:"
                    )
                    logger.info(f"Precision: {precision:.4f}")
                    logger.info(f"Recall: {recall:.4f}")
                    logger.info(f"F1-score: {f1_score_val:.4f}")
                    logger.info(f"Confusion Matrix:\n{conf_mat}")
                    logger.info(f"AUROC: {auroc:.4f}")

                    print(
                        f"Logistic Regression evaluation result for Trial {identifier}, Window Size {window_size}:"
                    )
                    print(f"Precision: {precision:.4f}")
                    print(f"Recall: {recall:.4f}")
                    print(f"F1-score: {f1_score_val:.4f}")
                    print("Confusion Matrix:")
                    print(conf_mat)
                    print(f"AUROC: {auroc:.4f}")

                    results_dir = os.path.join(output_dir, "results")
                    os.makedirs(results_dir, exist_ok=True)

                    results = {
                        "Precision": precision,
                        "Recall": recall,
                        "F1-score": f1_score_val,
                        "Confusion Matrix": conf_mat.tolist(),
                        "AUROC": auroc,
                    }

                    results_file = os.path.join(
                        results_dir,
                        f"lr_window_{window_size}_results_{identifier}.json",
                    )
                    with open(results_file, "w") as f:
                        json.dump(results, f, indent=4)
                    logger.info(f"Saved evaluation results to {results_file}")
                    print(f"Saved evaluation results to {results_file}")
                except Exception as e:
                    logger.error(
                        f"Error during evaluation metrics computation for Trial {identifier}, Window {window_size}: {e}",
                        exc_info=True,
                    )
                    print(
                        f"Error during evaluation metrics computation for Trial {identifier}, Window {window_size}: {e}"
                    )
                    continue

        else:
            raise ValueError("clf_mode must be either 'transformer' or 'lookback_ratio_lr'.")

    except Exception as e:
        traceback.print_exc()
        print(f"An error occurred: {e}")
    finally:
        print("Process finished")
        if "logger" in locals():
            logger.info("Process finished")

if __name__ == "__main__":
    main()
