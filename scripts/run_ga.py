from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.metrics import roc_auc_score
from pandas.api.types import is_numeric_dtype

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from algorithm.GA import (  # noqa: E402
    build_mlp_from_gene,
    generate_random_individual,
    mutate_individual,
    roulette_selection,
    single_point_crossover,
)
from src.utils import generate_run_id, seed_everything, setup_logging  # noqa: E402


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _encode_features(
    X_train: pd.DataFrame, X_val: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    combined = pd.concat([X_train, X_val], axis=0)
    for col in combined.columns:
        if is_numeric_dtype(combined[col]):
            combined[col] = combined[col].fillna(0)
        else:
            combined[col] = combined[col].fillna("unknown")
            combined[col], _ = pd.factorize(combined[col])
    combined = combined.astype(np.float32)
    X_train_enc = combined.iloc[: len(X_train)].values
    X_val_enc = combined.iloc[len(X_train) :].values
    return X_train_enc, X_val_enc


def _evaluate_individual(
    gene,
    input_dim: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    train_cfg: Dict,
    out_dir: str,
) -> float:
    model = build_mlp_from_gene(gene, input_dim)

    epochs = int(train_cfg.get("epochs", 3))
    batch_size = int(train_cfg.get("batch_size", 256))

    model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    preds = model.predict(X_val, batch_size=batch_size, verbose=0).ravel()
    try:
        auc = float(roc_auc_score(y_val, preds))
    except ValueError:
        auc = float("nan")

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gene.json"), "w", encoding="utf-8") as f:
        json.dump(gene.to_dict(), f, indent=2)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"val_roc_auc": auc}, f, indent=2)

    return auc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GA for MLP architecture search")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    tf.random.set_seed(seed)

    run_cfg = config.setdefault("run", {})
    if not run_cfg.get("run_id"):
        run_cfg["run_id"] = generate_run_id("ga")
    if "run_dir" not in run_cfg:
        output_dir = run_cfg.get("output_dir", "runs")
        run_cfg["run_dir"] = os.path.join(output_dir, run_cfg["run_id"])
    run_dir = run_cfg["run_dir"]

    logger = setup_logging(run_dir, name="nic.ga")

    from preprocessing.prepare_data import process_data  # noqa: E402

    data_cfg = config.get("data", {})
    def _resolve_path(p: str | None) -> str | None:
        if not p:
            return None
        return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)

    data = process_data(
        train_transaction_path=_resolve_path(data_cfg.get("train_transaction_path")),
        train_identity_path=_resolve_path(data_cfg.get("train_identity_path")),
    )
    X_train_df, X_val_df, y_train_s, y_val_s = data

    X_train, X_val = _encode_features(X_train_df, X_val_df)
    y_train = y_train_s.values.astype(np.float32)
    y_val = y_val_s.values.astype(np.float32)
    input_dim = X_train.shape[1]

    ga_cfg = config.get("ga", {})
    train_cfg = config.get("train", {})

    pop_size = int(ga_cfg.get("population_size", 5))
    generations = int(ga_cfg.get("generations", 2))
    elitism = int(ga_cfg.get("elitism", 1))
    selection = str(ga_cfg.get("selection", "roulette"))
    crossover_rate = float(ga_cfg.get("crossover_rate", 0.9))
    mutation_rate = float(ga_cfg.get("mutation_rate", 0.3))

    min_layers = int(ga_cfg.get("min_layers", 1))
    max_layers = int(ga_cfg.get("max_layers", 5))
    units_choices = list(ga_cfg.get("units_choices", [16, 32, 64, 128, 256]))
    activation_choices = list(ga_cfg.get("activation_choices", ["relu", "tanh", "sigmoid"]))
    dropout_range = tuple(ga_cfg.get("dropout_range", [0.0, 0.5]))

    population = [
        generate_random_individual(
            min_layers=min_layers,
            max_layers=max_layers,
            units_choices=units_choices,
            activation_choices=activation_choices,
            dropout_range=dropout_range,
        )
        for _ in range(pop_size)
    ]

    history = []
    best = {"fitness": float("-inf"), "gene": None}

    for gen in range(generations):
        logger.info(f"Generation {gen + 1}/{generations}")
        evaluated = []
        for idx, gene in enumerate(population):
            ind_dir = os.path.join(run_dir, f"gen_{gen+1}", f"ind_{idx+1}")
            fitness = _evaluate_individual(
                gene,
                input_dim,
                X_train,
                y_train,
                X_val,
                y_val,
                train_cfg,
                ind_dir,
            )
            evaluated.append((gene, fitness))
            logger.info(f"  Individual {idx+1}/{pop_size} fitness={fitness:.4f}")

        evaluated.sort(key=lambda x: x[1], reverse=True)
        gen_best_gene, gen_best_fitness = evaluated[0]

        if gen_best_fitness > best["fitness"]:
            best = {"fitness": gen_best_fitness, "gene": gen_best_gene.to_dict()}

        avg_fitness = float(np.mean([f for _, f in evaluated]))
        history.append(
            {
                "generation": gen + 1,
                "best_fitness": float(gen_best_fitness),
                "avg_fitness": avg_fitness,
            }
        )

        if gen == generations - 1:
            break

        next_population = [gene for gene, _ in evaluated[:elitism]]
        while len(next_population) < pop_size:
            if selection != "roulette":
                logger.info(
                    f"Selection '{selection}' not supported in GA.py. Using roulette."
                )
            parents = roulette_selection([(g, f, f) for g, f in evaluated], 2)

            if np.random.rand() < crossover_rate:
                child1, child2 = single_point_crossover(parents[0], parents[1])
            else:
                child1, child2 = parents[0], parents[1]

            child1 = mutate_individual(
                child1,
                mutation_rate=mutation_rate,
                units_choices=units_choices,
                activation_choices=activation_choices,
                dropout_range=dropout_range,
            )
            if len(next_population) < pop_size:
                next_population.append(child1)

            child2 = mutate_individual(
                child2,
                mutation_rate=mutation_rate,
                units_choices=units_choices,
                activation_choices=activation_choices,
                dropout_range=dropout_range,
            )
            if len(next_population) < pop_size:
                next_population.append(child2)

        population = next_population

    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump({"history": history}, f, indent=2)
    with open(os.path.join(run_dir, "best.json"), "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print("Best individual:", best)


if __name__ == "__main__":
    main()
