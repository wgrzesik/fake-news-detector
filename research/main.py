import os
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
from research.factory import ModelFactory
from research.embeddings.base_embedding import BaseEmbedder
import random

SEED = 42

DATASETS = ['ISOT', 'LIAR', 'WELFake']

ALL_MODELS_TO_TEST = [
    'svm', 
    'lr',
    'nb',
    'mnb',
    'knn',
    'rf',
    'dt',
    'xgb'
]

ALL_EMBEDDINGS_TO_TEST = [
    'tfidf', 
    'word2vec',
    'glove',
    'bow'
]

BASE_DATA_DIR = os.path.join('research', 'data', 'processed')

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def load_data(dataset_name: str, split_type: str):
    file_name = f"{split_type}.csv"
    file_path = os.path.join(BASE_DATA_DIR, dataset_name, file_name)

    try:
        if not os.path.exists(file_path):
            return [], []

        df = pd.read_csv(file_path)
        df = df.dropna(subset=['text', 'label'])
        
        texts = df['text'].astype(str).tolist()
        labels = df['label'].tolist()
        
        print(f"[Data] Loaded {len(texts)} rows from {dataset_name}/{file_name}")
        return texts, labels
    
    except Exception as e:
        print(f"[Error] Loading {file_path}: {e}")
        return [], []

def run_benchmark() -> None:
    set_seed(SEED)
    print(f"\n=== START BENCHMARK @ {datetime.now()} ===")

    for dataset in DATASETS:
        print(f"\n===== DATASET: {dataset} =====")

        X_train_raw, y_train = load_data(dataset, "train")
        X_test_raw, y_test = load_data(dataset, "test")

        if not X_train_raw or not X_test_raw:
            print("[SKIP] Missing data")
            continue

        dataset_results = []

        for embedding_type in ALL_EMBEDDINGS_TO_TEST:
            print(f"\n[Embedding] {embedding_type.upper()}")

            try:
                master_embedder = BaseEmbedder.create(embedding_type)

                t0 = datetime.now()
                X_train_vec = master_embedder.fit_transform(X_train_raw)
                X_test_vec = master_embedder.transform(X_test_raw)
                vec_time = (datetime.now() - t0).total_seconds()

                print(f"[Vectorized] {X_train_vec.shape} ({vec_time:.2f}s)")

                for model_type in ALL_MODELS_TO_TEST:
                    if embedding_type not in ModelFactory.get_valid_embeddings_for_model(model_type):
                        continue

                    print(f"[Run] {model_type}+{embedding_type} ...", end=" ", flush=True)

                    try:
                        model = ModelFactory.get_model(
                            dataset_name=dataset,
                            model_type=model_type,
                            embedding_type=embedding_type
                        )

                        model.embedder = master_embedder.clone()

                        t_train = datetime.now()
                        model.train_on_vectors(X_train_vec, y_train)
                        train_time = (datetime.now() - t_train).total_seconds()

                        metrics = model.evaluate_on_vectors(X_test_vec, y_test)
                        model.save()

                        dataset_results.append({
                            "dataset": dataset,
                            "model": model_type,
                            "embedding": embedding_type,
                            "accuracy": round(metrics["accuracy"], 4),
                            "precision": round(metrics["precision"], 4),
                            "recall": round(metrics["recall"], 4),
                            "f1": round(metrics["f1_score"], 4),
                            "train_time_sec": round(train_time, 2),
                            "timestamp": datetime.now().isoformat()
                        })

                        print("DONE")

                    except Exception:
                        print("FAILED")
                        traceback.print_exc()

            except Exception:
                print(f"[Embedding ERROR] {embedding_type}")
                traceback.print_exc()

        if dataset_results:
            out_file = f"results_{dataset}.csv"
            pd.DataFrame(dataset_results).to_csv(
                out_file,
                mode="a",
                header=not os.path.exists(out_file),
                index=False
            )
            print(f"[Saved] {len(dataset_results)} → {out_file}")

    print(f"\n=== BENCHMARK END @ {datetime.now()} ===")


if __name__ == "__main__":
    run_benchmark()