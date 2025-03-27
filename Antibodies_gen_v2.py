#!/usr/bin/env python
"""
Comprehensive Finetuning, Structure Prediction, and Docking Pipeline for DeepSeek-R1 1.5B on Antibody Sequences

This script implements a full pipeline with the following steps:
    1. Load a large database of antibody sequences with annotated properties.
    2. Validate and split the data into training, validation, and test sets.
    3. Finetune the DeepSeek-R1 1.5B model (DeepSeek-R1-Distill-Qwen-1.5B) on the training set using Hugging Face’s Trainer API.
    4. Evaluate the model on validation and test sets and generate extensive plots.
    5. Generate new antibody sequences using the finetuned model.
    6. For each generated sequence, predict its 3D structure using ColabFold (an AlphaFold2 pipeline) and then dock the predicted structure to a reference protein using idock.
    7. Save all outputs, including generated sequences, predicted structures, docking binding energies, and plots.

This code is heavily instrumented with assertions, debugging, and logging for maximum traceability.

Author: Your Name
Date: 2025-02-22
"""

import os
import sys
import json
import random
import argparse
import logging
import subprocess
import shutil
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import requests
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import re

# Hugging Face libraries for model finetuning
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    PreTrainedModel, PreTrainedTokenizer, DataCollatorForLanguageModeling
)

# ColabFold imports for structure prediction using AlphaFold2 pipeline
from colabfold.download import download_alphafold_params, default_data_dir
from colabfold.batch import get_queries, run, set_model_type

# For plotting correlations and advanced visualizations
import seaborn as sns

# =============================================================================
# Utility Functions
# =============================================================================

def reconstruct_sequence(chain_list: List[List]) -> str:
    """
    Given a list where each element is in the format [[position, ' '], 'A'],
    reconstruct the sequence string by concatenating the second element of each sub‑list.
    """
    return "".join([item[1] for item in chain_list])

def clean_sequence(seq: str) -> str:
    """
    Clean the generated sequence so that only valid amino acid letters remain.
    Valid amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y.
    """
    seq = str(seq).upper().strip()
    # Remove any character not among the allowed amino acids.
    return re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', seq)

def write_sequence_csv(csv_path: str, jobname: str, sequence: str) -> None:
    """
    Write the sequence CSV file using proper quoting to avoid delimiter issues.
    """
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "sequence"])
        writer.writerow([jobname, sequence])

def process_json_antibody_data(filepath: str) -> List[Dict[str, Any]]:
    """
    Load the JSON file and pair heavy and light chains.
    """
    print(f"Loading JSON file from '{filepath}'...")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    paired_antibodies = {}
    for key, chain_data in data.items():
        parts = key.split('_')
        if len(parts) != 2:
            continue
        antibody_id, chain_type = parts
        if antibody_id not in paired_antibodies:
            paired_antibodies[antibody_id] = {}
        sequence = reconstruct_sequence(chain_data)
        paired_antibodies[antibody_id][chain_type] = sequence
    antibody_records = []
    for antibody_id, chains in paired_antibodies.items():
        heavy_seq = chains.get("VH")
        light_seq = chains.get("VL")
        if heavy_seq and light_seq:
            combined_seq = heavy_seq + " [SEP] " + light_seq
        elif heavy_seq:
            combined_seq = heavy_seq
        elif light_seq:
            combined_seq = light_seq
        else:
            continue
        record = {
            "antibody_id": antibody_id,
            "heavy_sequence": heavy_seq,
            "light_sequence": light_seq,
            "sequence": combined_seq,
            "binding_affinity": 0.0,
            "stability": 0.0,
            "solubility": 0.0,
            "other_properties": {}
        }
        antibody_records.append(record)
    print(f"Processed {len(antibody_records)} antibody records from the JSON file.")
    return antibody_records

def load_antibody_data(filepath: str) -> List[Dict[str, Any]]:
    """
    Load antibody data from a CSV file.
    """
    logger.info(f"Loading antibody data from '{filepath}'...")
    assert os.path.exists(filepath), f"Data file {filepath} does not exist."
    try:
        df = pd.read_csv(filepath, encoding="utf-8-sig", sep=",")
        required_columns = ["Name", "VHorVHH", "VL"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV must contain '{col}' column.")
        records = []
        for _, row in df.iterrows():
            antibody_id = str(row["Name"]).strip()
            heavy_seq = str(row["VHorVHH"]).strip() if pd.notna(row["VHorVHH"]) else ""
            light_seq = str(row["VL"]).strip() if pd.notna(row["VL"]) else ""
            if heavy_seq and light_seq:
                combined_seq = heavy_seq + " [SEP] " + light_seq
            elif heavy_seq:
                combined_seq = heavy_seq
            elif light_seq:
                combined_seq = light_seq
            else:
                continue
            record = {
                "antibody_id": antibody_id,
                "heavy_sequence": heavy_seq,
                "light_sequence": light_seq,
                "sequence": combined_seq,
                "binding_affinity": 0.0,
                "stability": 0.0,
                "solubility": 0.0,
                "other_properties": {}
            }
            records.append(record)
        logger.info(f"Loaded {len(records)} records from {filepath}.")
        return records
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def split_data(data: List[Dict[str, Any]], train_ratio: float = 0.8, val_ratio: float = 0.1) -> (List, List, List):
    """
    Split data into training, validation, and test sets.
    """
    logger.info("Splitting data into train/val/test sets...")
    total = len(data)
    assert total > 0, "No data available for splitting."
    random.shuffle(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    logger.info("Data split: {} training, {} validation, {} test samples.".format(len(train_data), len(val_data), len(test_data)))
    return train_data, val_data, test_data

# =============================================================================
# Dataset Definition
# =============================================================================

class AntibodyDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for antibody sequences with annotations.
    """
    def __init__(self, data: List[Dict[str, Any]], tokenizer: PreTrainedTokenizer, max_length: int = 512):
        logger.info("Initializing AntibodyDataset with {} samples.".format(len(data)))
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._validate_data()
        logger.info("Dataset successfully validated.")

    def _validate_data(self) -> None:
        logger.debug("Validating dataset records...")
        for idx, record in enumerate(self.data):
            assert 'sequence' in record, f"Record {idx} missing 'sequence'."
            assert isinstance(record['sequence'], str) and record['sequence'].strip(), f"Record {idx} has invalid 'sequence'."
            assert 'binding_affinity' in record, f"Record {idx} missing 'binding_affinity'."
            assert isinstance(record['binding_affinity'], (float, int)), f"Record {idx} 'binding_affinity' is not numeric."
            assert 'stability' in record, f"Record {idx} missing 'stability'."
            assert isinstance(record['stability'], (float, int)), f"Record {idx} 'stability' is not numeric."
            assert 'solubility' in record, f"Record {idx} missing 'solubility'."
            assert isinstance(record['solubility'], (float, int)), f"Record {idx} 'solubility' is not numeric."
        logger.debug("All records validated successfully.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.data[idx]
        sequence = record['sequence']
        tokenized = self.tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        input_ids = tokenized["input_ids"].squeeze()
        attention_mask = tokenized["attention_mask"].squeeze()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "binding_affinity": float(record['binding_affinity']),
            "stability": float(record['stability']),
            "solubility": float(record['solubility']),
            "other_properties": record.get("other_properties", {}),
            "raw_sequence": sequence
        }

# =============================================================================
# Model and Tokenizer Loading
# =============================================================================

def load_model_and_tokenizer(model_name: str) -> (PreTrainedModel, PreTrainedTokenizer):
    logger.info("Loading model and tokenizer for '{}'...".format(model_name))
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        logger.info("Model and tokenizer loaded successfully.")
    except Exception as e:
        logger.error("Error loading model/tokenizer: {}".format(e))
        raise
    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        logger.debug("Pad token not found or same as EOS token. Adding a distinct pad token.")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer

# =============================================================================
# Training Functionality Using Hugging Face Trainer
# =============================================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # Convert numpy arrays to tensors
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    labels_flat = labels.reshape(-1)
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_fct(logits_flat, labels_flat)
    return {"loss": loss.item()}

def train_model(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, train_dataset: torch.utils.data.Dataset,
                val_dataset: torch.utils.data.Dataset, output_dir: str, num_train_epochs: int = 3,
                per_device_train_batch_size: int = 4, per_device_eval_batch_size: int = 4,
                learning_rate: float = 5e-5, weight_decay: float = 0.01, logging_steps: int = 50,
                save_steps: int = 200, max_steps: int = -1):
    logger.info("Starting model training...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        fp16=True,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_steps=save_steps,
        max_steps=max_steps if max_steps > 0 else -1,
        evaluation_strategy="steps",
        eval_steps=save_steps,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to=[],
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )
    logger.debug("Training arguments set.")
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    logger.debug("Data collator initialized.")
    assert len(train_dataset) > 0, "Training dataset is empty!"
    assert len(val_dataset) > 0, "Validation dataset is empty!"
    logger.debug("Datasets confirmed non-empty.")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    logger.info("Trainer initialized. Beginning training...")
    train_result = trainer.train()
    logger.info("Training complete.")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model and tokenizer saved to {}.".format(output_dir))
    for key, value in train_result.metrics.items():
        logger.info("Train metric - {}: {}".format(key, value))
    eval_result = trainer.evaluate()
    for key, value in eval_result.items():
        logger.info("Validation metric - {}: {}".format(key, value))
    return trainer, train_result.metrics, eval_result

# =============================================================================
# Plotting Functions
# =============================================================================

def plot_sequence_length_distribution(dataset: torch.utils.data.Dataset, output_path: str) -> None:
    logger.info("Plotting sequence length distribution...")
    lengths = [len(sample["raw_sequence"]) for sample in dataset]
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, color='skyblue', edgecolor='black')
    plt.xlabel("Sequence Length (amino acids)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Antibody Sequence Lengths")
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    logger.info("Sequence length plot saved to {}.".format(output_path))

def plot_annotated_property_distribution(dataset: torch.utils.data.Dataset, property_name: str,
                                         output_path: str) -> None:
    logger.info("Plotting distribution for property '{}'...".format(property_name))
    values = [sample[property_name] for sample in dataset if property_name in sample]
    assert len(values) > 0, "No values found for property '{}'.".format(property_name)
    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=50, color='salmon', edgecolor='black')
    plt.xlabel(property_name.replace('_', ' ').capitalize())
    plt.ylabel("Frequency")
    plt.title("Distribution of {}".format(property_name.replace('_', ' ').capitalize()))
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    logger.info("Property distribution plot saved to {}.".format(output_path))

def plot_training_loss(metrics: Dict[str, Any], output_path: str) -> None:
    logger.info("Plotting training loss curve...")
    if "loss" in metrics:
        loss_values = metrics["loss"]
        steps = list(range(len(loss_values)))
        plt.figure(figsize=(10, 6))
        plt.plot(steps, loss_values, label="Training Loss", color='blue')
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss over Steps")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_path)
        plt.close()
        logger.info("Training loss plot saved to {}.".format(output_path))
    else:
        logger.warning("No training loss data available for plotting.")

def plot_property_correlations(dataset: torch.utils.data.Dataset, properties: List[str], output_path: str) -> None:
    logger.info("Plotting property correlations...")
    data_dict = {prop: [] for prop in properties}
    for sample in dataset:
        for prop in properties:
            data_dict[prop].append(sample.get(prop, None))
    df = pd.DataFrame(data_dict)
    sns.pairplot(df)
    plt.suptitle("Correlations between Annotated Properties", y=1.02)
    plt.savefig(output_path)
    plt.close()
    logger.info("Scatter plot matrix saved to {}.".format(output_path))

def additional_plots_for_annotated_properties(dataset: torch.utils.data.Dataset, output_dir: str) -> None:
    logger.info("Generating additional plots for annotated properties...")
    records = []
    for sample in dataset:
        record = {
            "binding_affinity": sample["binding_affinity"],
            "stability": sample["stability"],
            "solubility": sample["solubility"]
        }
        for key, value in sample["other_properties"].items():
            record[key] = value
        records.append(record)
    df = pd.DataFrame(records)
    # Box plot
    plt.figure(figsize=(12, 8))
    df.boxplot()
    plt.title("Box Plot of Annotated Properties")
    plt.ylabel("Value")
    box_plot_path = os.path.join(output_dir, "annotated_properties_boxplot.png")
    plt.savefig(box_plot_path)
    plt.close()
    logger.info("Box plot saved to {}.".format(box_plot_path))
    # Violin plots per property
    for column in df.columns:
        plt.figure(figsize=(10, 6))
        sns.violinplot(y=df[column])
        plt.title("Violin Plot of {}".format(column.replace('_', ' ').capitalize()))
        plt.ylabel("Value")
        violin_plot_path = os.path.join(output_dir, "{}_violin_plot.png".format(column))
        plt.savefig(violin_plot_path)
        plt.close()
        logger.info("Violin plot for {} saved to {}.".format(column, violin_plot_path))
    # Scatter matrix
    sns.pairplot(df)
    plt.suptitle("Scatter Plot Matrix of Annotated Properties", y=1.02)
    scatter_plot_path = os.path.join(output_dir, "annotated_properties_scatter_matrix.png")
    plt.savefig(scatter_plot_path)
    plt.close()
    logger.info("Scatter matrix saved to {}.".format(scatter_plot_path))

# =============================================================================
# Debugging and Summary Utilities
# =============================================================================

def debug_print_model_info(model: PreTrainedModel) -> None:
    logger.info("Detailed Model Information:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total parameters: {}".format(total_params))
    logger.info("Trainable parameters: {}".format(trainable_params))
    for name, param in model.named_parameters():
        logger.debug("Parameter: {} | Shape: {} | Trainable: {} | Device: {}".format(name, param.shape, param.requires_grad, param.device))

def debug_print_tokenizer_info(tokenizer: PreTrainedTokenizer) -> None:
    logger.info("Tokenizer Information:")
    vocab_size = len(tokenizer)
    pad_token = tokenizer.pad_token if tokenizer.pad_token is not None else "None"
    eos_token = tokenizer.eos_token if tokenizer.eos_token is not None else "None"
    bos_token = tokenizer.bos_token if tokenizer.bos_token is not None else "None"
    logger.info("Vocabulary size: {}".format(vocab_size))
    logger.info("Pad token: {}".format(pad_token))
    logger.info("EOS token: {}".format(eos_token))
    logger.info("BOS token: {}".format(bos_token))

def assert_model_is_on_correct_device(model: PreTrainedModel, device: torch.device) -> None:
    logger.info("Verifying model device allocation...")
    for name, param in model.named_parameters():
        assert param.device.type == device.type, "Parameter {} is on {} instead of {}.".format(name, param.device, device)
    logger.info("All model parameters are correctly allocated to {}.".format(device))

def print_hyperparameters(args: argparse.Namespace) -> None:
    logger.info("Training hyperparameters:")
    for arg in vars(args):
        logger.info("{}: {}".format(arg, getattr(args, arg)))

def print_dataset_summary(dataset: torch.utils.data.Dataset) -> None:
    logger.info("Dataset Summary:")
    num_samples = len(dataset)
    sequences = [sample["raw_sequence"] for sample in dataset]
    lengths = [len(seq) for seq in sequences]
    avg_length = np.mean(lengths)
    min_length = np.min(lengths)
    max_length = np.max(lengths)
    logger.info("Total samples: {}".format(num_samples))
    logger.info("Sequence Length - Average: {:.2f}, Min: {}, Max: {}".format(avg_length, min_length, max_length))
    for i in range(min(5, num_samples)):
        logger.debug("Sample {}: {}... | Binding Affinity: {}, Stability: {}, Solubility: {}".format(
            i + 1, dataset[i]["raw_sequence"][:50], dataset[i]["binding_affinity"],
            dataset[i]["stability"], dataset[i]["solubility"]
        ))

def save_dataset_statistics(dataset: torch.utils.data.Dataset, output_path: str) -> None:
    logger.info("Saving dataset statistics to {}...".format(output_path))
    num_samples = len(dataset)
    sequences = [sample["raw_sequence"] for sample in dataset]
    lengths = [len(seq) for seq in sequences]
    stats = {
        "total_samples": num_samples,
        "average_length": float(np.mean(lengths)),
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths))
    }
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=4)
        logger.info("Dataset statistics saved successfully.")
    except Exception as e:
        logger.error("Failed to save dataset statistics: {}".format(e))

# =============================================================================
# 3D Structure Prediction with ColabFold
# =============================================================================

def predict_structure_with_colabfold(sequence: str, result_dir: str, jobname: str,
                                     msa_mode: str = "single_sequence", num_relax: int = 0,
                                     model_type: str = "auto") -> str:
    logger.info("Predicting 3D structure for job '{}'...".format(jobname))
    os.makedirs(result_dir, exist_ok=True)
    csv_path = os.path.join(result_dir, "{}.csv".format(jobname))
    # Clean and ensure sequence is a string
    sequence = clean_sequence(sequence)
    write_sequence_csv(csv_path, jobname, sequence)
    logger.debug("Wrote sequence CSV to {}.".format(csv_path))
    queries, is_complex = get_queries(csv_path)
    final_model_type = set_model_type(is_complex, model_type)
    download_alphafold_params(final_model_type, Path("."))
    results = run(
        queries=queries,
        result_dir=result_dir,
        use_templates=False,
        num_relax=num_relax,
        msa_mode=msa_mode,
        model_type=final_model_type,
        num_models=5,
        num_recycles=3,
        relax_max_iterations=200,
        recycle_early_stop_tolerance=0.0,
        num_seeds=1,
        use_dropout=False,
        model_order=[1, 2, 3, 4, 5],
        is_complex=is_complex,
        data_dir=Path("."),
        keep_existing_results=False,
        rank_by="auto",
        pair_mode="unpaired",
        pairing_strategy="greedy",
        stop_at_score=100.0,
        dpi=200,
        zip_results=False
    )
    pdb_file = os.path.join(result_dir, "ranked_0.pdb")
    if not os.path.exists(pdb_file):
        for root, _, files in os.walk(result_dir):
            for file in files:
                if file.endswith(".pdb"):
                    pdb_file = os.path.join(root, file)
                    break
            if os.path.exists(pdb_file):
                break
    if not os.path.exists(pdb_file):
        raise FileNotFoundError("ColabFold did not produce a PDB file for job '{}'.".format(jobname))
    logger.info("Predicted structure saved to {}.".format(pdb_file))
    return pdb_file

# =============================================================================
# Docking Functionality Using idock
# =============================================================================

def perform_docking(ligand_pdb: str, receptor_pdbqt: str, out_dir: str,
                    center: list, box_size: list, exhaustiveness: int = 32, n_poses: int = 20) -> float:
    logger.info("Performing docking for ligand '{}' against receptor '{}'...".format(ligand_pdb, receptor_pdbqt))
    docking_dir = os.path.join(out_dir, "docking_results")
    os.makedirs(docking_dir, exist_ok=True)
    ligand_pdbqt = os.path.join(docking_dir, "ligand.pdbqt")
    try:
        subprocess.run(["obabel", ligand_pdb, "-O", ligand_pdbqt], check=True)
    except subprocess.CalledProcessError as e:
        logger.error("Error converting ligand PDB to PDBQT: {}".format(e))
        raise
    config_file = os.path.join(docking_dir, "idock_config.txt")
    config_contents = (
        f"receptor = {receptor_pdbqt}\n"
        f"ligand = {ligand_pdbqt}\n"
        f"center_x = {center[0]}\n"
        f"center_y = {center[1]}\n"
        f"center_z = {center[2]}\n"
        f"size_x = {box_size[0]}\n"
        f"size_y = {box_size[1]}\n"
        f"size_z = {box_size[2]}\n"
        f"num_modes = {n_poses}\n"
        f"exhaustiveness = {exhaustiveness}\n"
    )
    with open(config_file, "w") as f:
        f.write(config_contents)
    logger.debug("Wrote idock configuration to {}.".format(config_file))
    idock_output = os.path.join(docking_dir, "docked_out.txt")
    try:
        subprocess.run(["idock", "--config", config_file, "--out", idock_output], check=True)
    except subprocess.CalledProcessError as e:
        logger.error("Error running idock: {}".format(e))
        raise
    best_energy = None
    try:
        with open(idock_output, "r") as f:
            for line in f:
                if "affinity" in line.lower():
                    parts = line.split()
                    for token in parts:
                        try:
                            value = float(token)
                            best_energy = value
                            break
                        except ValueError:
                            continue
                    if best_energy is not None:
                        break
        if best_energy is None:
            raise ValueError("No binding energy found in idock output.")
    except Exception as e:
        logger.error("Error parsing idock output: {}".format(e))
        raise
    logger.info("Docking completed. Best binding energy: {:.2f} kcal/mol".format(best_energy))
    return best_energy

# =============================================================================
# Extended Main Pipeline
# =============================================================================

def extended_main():
    logger.info("Starting Extended Finetuning, Structure Prediction, and Docking Pipeline...")
    parser = argparse.ArgumentParser(
        description="Extended Pipeline for Finetuning DeepSeek-R1 1.5B, 3D Structure Prediction, and Docking on Antibody Data"
    )
    parser.add_argument("--data_file", type=str, required=True,
                        help="Path to JSON or CSV file containing antibody sequences with annotations.")
    parser.add_argument("--output_dir", type=str, default="finetune_output",
                        help="Directory to save outputs, models, and plots.")
    parser.add_argument("--model_name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", #Qwen/Qwen2.5-0.5B
                        help="Pretrained model identifier (e.g., deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)")#Qwen/Qwen2.5-0.5B
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum tokenized sequence length.")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,
                        help="Batch size per device during training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay.")
    parser.add_argument("--logging_steps", type=int, default=50,
                        help="Logging interval (steps).")
    parser.add_argument("--save_steps", type=int, default=200,
                        help="Checkpoint saving interval (steps).")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Maximum training steps (-1 for no limit).")
    # Additional parameters for structure prediction and docking
    parser.add_argument("--msa_mode", type=str, default="single_sequence",
                        help="MSA mode for ColabFold (e.g., 'single_sequence').")
    parser.add_argument("--num_relax", type=int, default=0,
                        help="Number of relaxation steps for ColabFold.")
    parser.add_argument("--af_model_type", type=str, default="auto",
                        help="Model type for ColabFold (e.g., 'auto').")
    parser.add_argument("--reference_protein", type=str, required=True,
                        help="Path to the reference receptor protein in PDBQT format for docking.")
    parser.add_argument("--docking_center", type=float, nargs=3, default=[0, 0, 0],
                        help="Docking box center coordinates [x, y, z].")
    parser.add_argument("--docking_box", type=float, nargs=3, default=[20, 20, 20],
                        help="Docking box size [x, y, z].")
    parser.add_argument("--exhaustiveness", type=int, default=32,
                        help="Exhaustiveness for docking search.")
    parser.add_argument("--n_poses", type=int, default=20,
                        help="Number of docking poses to generate.")
    args = parser.parse_args()

    print_hyperparameters(args)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Output directory: {}".format(args.output_dir))

    # Load and split dataset
    if args.data_file.endswith(".json"):
        data = process_json_antibody_data(args.data_file)
    else:
        data = load_antibody_data(args.data_file)
    train_data, val_data, test_data = split_data(data)
    logger.info("Data loaded and split into training, validation, and test sets.")

    # Load model and tokenizer; move model to correct device
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info("Model moved to device: {}".format(device))
    assert_model_is_on_correct_device(model, device)
    debug_print_model_info(model)
    debug_print_tokenizer_info(tokenizer)

    # Create dataset objects (using subsampled data for quicker runs)
    small_train_data = train_data#[:1000]
    small_val_data = val_data#[:100]
    small_test_data = test_data#[:100]
    train_dataset = AntibodyDataset(small_train_data, tokenizer, max_length=args.max_length)
    val_dataset = AntibodyDataset(small_val_data, tokenizer, max_length=args.max_length)
    test_dataset = AntibodyDataset(small_test_data, tokenizer, max_length=args.max_length)
    print_dataset_summary(train_dataset)
    print_dataset_summary(val_dataset)
    print_dataset_summary(test_dataset)
    save_dataset_statistics(train_dataset, os.path.join(args.output_dir, "train_stats.json"))
    save_dataset_statistics(val_dataset, os.path.join(args.output_dir, "val_stats.json"))
    save_dataset_statistics(test_dataset, os.path.join(args.output_dir, "test_stats.json"))

    # Generate diagnostic plots
    plot_sequence_length_distribution(train_dataset, os.path.join(args.output_dir, "train_sequence_lengths.png"))
    plot_annotated_property_distribution(train_dataset, "binding_affinity", os.path.join(args.output_dir, "train_binding_affinity.png"))
    plot_annotated_property_distribution(train_dataset, "stability", os.path.join(args.output_dir, "train_stability.png"))
    plot_annotated_property_distribution(train_dataset, "solubility", os.path.join(args.output_dir, "train_solubility.png"))
    plot_property_correlations(train_dataset, ["binding_affinity", "stability", "solubility"],
                               os.path.join(args.output_dir, "train_property_correlations.png"))
    additional_plots_for_annotated_properties(train_dataset, args.output_dir)

    # Set GPU memory fraction before training (if using CUDA)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9, 0)

    # Finetune the model
    trainer, train_metrics, eval_metrics = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        max_steps=args.max_steps
    )
    if "loss" in train_metrics:
        plot_training_loss(train_metrics, os.path.join(args.output_dir, "training_loss.png"))
    else:
        logger.warning("No training loss data available for plotting.")

    # Empty GPU cache and re-set memory fraction after training/evaluation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.9, 0)

    logger.info("Evaluating finetuned model on test dataset...")
    test_results = trainer.evaluate(test_dataset)
    for key, value in test_results.items():
        logger.info("Test {}: {}".format(key, value))

    # Generate new antibody sequences using the finetuned model.
    logger.info("Generating new antibody sequences...")
    generation_prompt = ("Generate a valid antibody amino acid sequence "
                         "(using only letters A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y): ")
    input_ids = tokenizer.encode(generation_prompt, return_tensors="pt").to(device)
    # Create an attention mask explicitly
    attention_mask = torch.ones_like(input_ids)
    max_gen_length = args.max_length  # use max_length as total length
    with torch.no_grad():
        generated_outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_gen_length,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            num_return_sequences=5,
            pad_token_id=tokenizer.pad_token_id
        )
    generated_sequences = []
    for output in generated_outputs:
        decoded = tokenizer.decode(output, skip_special_tokens=True)
        raw_seq = decoded.replace(generation_prompt, "").strip()
        seq = clean_sequence(raw_seq)
        generated_sequences.append(seq)
        logger.info("Generated sequence preview: {}...".format(seq[:50]))
    generated_file = os.path.join(args.output_dir, "generated_antibodies.txt")
    with open(generated_file, "w", encoding="utf-8") as f:
        for i, seq in enumerate(generated_sequences, start=1):
            f.write(">Generated_Antibody_{}\n{}\n\n".format(i, seq))
    logger.info("Generated antibody sequences saved to {}.".format(generated_file))

    # For each generated sequence, predict 3D structure and perform docking.
    docking_results_file = os.path.join(args.output_dir, "docking_results.txt")
    with open(docking_results_file, "w", encoding="utf-8") as dock_f:
        for seq in generated_sequences:
            jobname = "job_{}".format(abs(hash(seq)) % 100000)
            # Validate the generated sequence: skip if empty or not purely alphabetic.
            if not seq or not seq.isalpha():
                logger.warning("Skipping job {} due to invalid sequence: {}".format(jobname, seq))
                dock_f.write("Sequence {}: Invalid generated sequence.\n".format(jobname))
                continue
            try:
                logger.info("Processing sequence {} for structure prediction and docking...".format(jobname))
                pdb_file = predict_structure_with_colabfold(
                    sequence=seq,
                    result_dir=args.output_dir,
                    jobname=jobname,
                    msa_mode=args.msa_mode,
                    num_relax=args.num_relax,
                    model_type=args.af_model_type
                )
                binding_energy = perform_docking(
                    ligand_pdb=pdb_file,
                    receptor_pdbqt=args.reference_protein,
                    out_dir=args.output_dir,
                    center=args.docking_center,
                    box_size=args.docking_box,
                    exhaustiveness=args.exhaustiveness,
                    n_poses=args.n_poses
                )
                dock_f.write("Sequence {}: Binding Energy = {:.2f} kcal/mol\n".format(jobname, binding_energy))
                logger.info("Sequence {} processed. Docking binding energy: {:.2f} kcal/mol".format(jobname, binding_energy))
            except Exception as e:
                logger.error("Error processing sequence {}: {}".format(jobname, e))
                dock_f.write("Sequence {}: Error during processing: {}\n".format(jobname, e))
    logger.info("Docking results saved to {}.".format(docking_results_file))
    logger.info("Extended pipeline completed successfully.")

# =============================================================================
# Entry Point and Logger Setup
# =============================================================================

if __name__ == "__main__":
    # Configure logging to output to both console and a log file.
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("finetune_deepseek.log")
        ]
    )
    logger = logging.getLogger(__name__)
    logger.debug("Logger configured successfully.")

    # Set random seed for reproducibility.
    def set_seed(seed: int = 42) -> None:
        print("Setting random seed for reproducibility...")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print("Seed has been set to:", seed)
    set_seed(42)

    extended_main()
