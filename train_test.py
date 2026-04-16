import argparse
import random
import tqdm
import os
import yaml

import lightning as L
import numpy as np
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from sklearn.metrics import average_precision_score, roc_auc_score

from datasets import load_data
from mlp import AVH_Sup

def _materialize_paths(callbacks_cfg, project_name: str):
    """
    Generate project-specific log and checkpoint directories based on configuration.
    """
    logger_root = callbacks_cfg["logger"]["log_path"]
    ckpt_root   = callbacks_cfg["ckpt_args"]["ckpt_dir"]

    log_dir  = os.path.join(logger_root, project_name)
    ckpt_dir = os.path.join(ckpt_root, project_name)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    return log_dir, ckpt_dir

def init_callbacks(callbacks_cfg):
    # Initialize logger
    if callbacks_cfg["logger"]["name"] == "tensorboard":
        logger = TensorBoardLogger(save_dir=callbacks_cfg["logger"]["log_path"], name="")
    elif callbacks_cfg["logger"]["name"] == "csv":
        logger = CSVLogger(save_dir=callbacks_cfg["logger"]["log_path"], name="")
    else:
        raise ValueError(callbacks_cfg["logger"]["name"] + " not yet implemented!")

    callbacks = []
    
    # Initialize checkpointing
    if "ckpt_args" in callbacks_cfg:
        ck = callbacks_cfg["ckpt_args"]
        callbacks.append(
            ModelCheckpoint(
                monitor=ck.get("metric", None),
                mode=ck.get("mode", "min"),
                dirpath=ck["ckpt_dir"],
                filename="model-{epoch:02d}",
                save_top_k=ck.get("save_top_k", 3),
                save_last=True,
                every_n_epochs=ck.get("every_n_epochs", 1),
                auto_insert_metric_name=False
            )
        )

    # Initialize early stopping
    if "early_stopping" in callbacks_cfg:
        es = callbacks_cfg["early_stopping"]
        callbacks.append(
            EarlyStopping(
                monitor=es["metric"],
                mode=es["mode"],
                patience=es["patience"]
            )
        )
        
    return logger, callbacks


def set_seed(seed):
    print(f"Using seed: {seed}", flush=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(config):
    # Determine project name for organizing logs and checkpoints
    project_name = config.get(
        "project_name",
        f'{config["data_info"]["name"]}-{config["model_hparams"]["model_type"]}-seed{config["seed"]}'
    )

    # Setup project-specific directories
    cb = {**config["callbacks"]}
    cb["logger"] = {**cb["logger"]}
    cb["ckpt_args"] = {**cb["ckpt_args"]}

    cb["logger"]["log_path"] = os.path.join(cb["logger"]["log_path"], project_name)
    cb["ckpt_args"]["ckpt_dir"] = os.path.join(cb["ckpt_args"]["ckpt_dir"], project_name)
    os.makedirs(cb["logger"]["log_path"], exist_ok=True)
    os.makedirs(cb["ckpt_args"]["ckpt_dir"], exist_ok=True)

    # Initialize data and model
    train_dl, val_dl = load_data(config=config["data_info"])
    model = AVH_Sup(config=config)

    logger, callbacks = init_callbacks(cb)
    trainer = L.Trainer(max_epochs=config["epochs"], logger=logger, callbacks=callbacks)

    # Handle checkpoint resuming
    resume_ckpt = config.get("resume_ckpt", None)
    if resume_ckpt is None and config.get("resume_last", False):
        resume_ckpt = os.path.join(cb["ckpt_args"]["ckpt_dir"], "last.ckpt")

    trainer.fit(
        model=model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
        ckpt_path=resume_ckpt
    )



def test(config):
    test_dl = load_data(config=config["data_info"], test=True)
    model = AVH_Sup.load_from_checkpoint(config["ckpt_path"])

    model.to("cuda")
    model.eval()

    all_scores = np.array([])
    all_labels = np.array([])
    all_paths = np.array([])
    
    with torch.no_grad():
        for batch in tqdm.tqdm(test_dl):
            video_feats, audio_feats, labels, paths = batch
            video_feats, audio_feats = video_feats.to("cuda"), audio_feats.to("cuda")

            scores = model.predict_scores(video_feats, audio_feats)

            all_scores = np.concatenate((all_scores, scores.cpu().numpy()), axis=0)
            all_labels = np.concatenate((all_labels, labels.cpu().numpy()), axis=0)
            all_paths = np.concatenate((all_paths, paths), axis=0)

    os.makedirs(config["output_path"], exist_ok=True)

    pd.DataFrame({
        "paths": all_paths,
        "scores": all_scores,
        "labels": all_labels
    }).to_csv(os.path.join(config["output_path"], "results.csv"), index=False)

    vals, cnts = np.unique(all_labels, return_counts=True)
    print("Label distribution:", dict(zip(vals.tolist(), cnts.tolist())))

    with open(os.path.join(config["output_path"], "eval_results.txt"), "w") as f:
        f.write(f"AUC: {roc_auc_score(y_score=all_scores, y_true=all_labels)}\n")
        f.write(f"AP: {average_precision_score(y_score=all_scores, y_true=all_labels)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Training and testing loop for Audio-Visual models'
    )

    parser.add_argument('--config_path', default='configs/test_config.yaml')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"])
    
    if args.test:
        test(config=config)
    else:
        train(config=config)
