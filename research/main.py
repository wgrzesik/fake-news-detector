import os
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
from research.factory import ModelFactory
from research.embeddings.base_embedding import BaseEmbedder

DATASETS = ['ISOT', 'LIAR', 'WELFake']

ALL_MODELS_TO_TEST = [
    'svm', 
    'lr',
    'nb',
    'mnb'
]

ALL_EMBEDDINGS_TO_TEST = [
    'tfidf', 
    'word2vec',
    'glove',
    'bow'
]

BASE_DATA_DIR = os.path.join('research', 'data', 'processed')

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

def run_benchmark():
    print(f"=== STARTING RESEARCH BENCHMARK: {datetime.now()} ===")

    for dataset in DATASETS:
        dataset_results = []
        RESULTS_FILE = f"results_{dataset}.csv"
        
        print(f"\nPROCESSING DATASET: {dataset.upper()}")
        
        raw_train_texts, y_train = load_data(dataset, 'train')
        raw_test_texts, y_test = load_data(dataset, 'test')
        
        if not raw_train_texts or not raw_test_texts:
            print(f"[SKIP] Missing data for {dataset}")
            continue

        print(f"[Clean] Preprocessing {len(raw_train_texts) + len(raw_test_texts)} articles")
        X_train_clean = [BaseEmbedder.preprocess_text(t) for t in raw_train_texts]
        X_test_clean = [BaseEmbedder.preprocess_text(t) for t in raw_test_texts]

        for embedding_type in ALL_EMBEDDINGS_TO_TEST:
            print(f"\nVectorizing with {embedding_type.upper()}")
            
            try:
                master_embedder = BaseEmbedder.create(embedding_type)
                
                start_vec = datetime.now()
                X_train_vec = master_embedder.fit_transform(X_train_clean)
                X_test_vec = master_embedder.transform(X_test_clean)
                vec_time = (datetime.now() - start_vec).total_seconds()
                print(f"[Vectorized] Matrix shape: {X_train_vec.shape} ({vec_time:.2f}s)")

                for model_type in ALL_MODELS_TO_TEST:
                    valid_embeddings = ModelFactory.get_valid_embeddings_for_model(model_type)
                    if embedding_type not in valid_embeddings:
                        continue

                    config_id = f"{model_type}+{embedding_type}"
                    print(f"[RUNNING] {config_id}...", end=" ", flush=True)

                    try:
                        model = ModelFactory.get_model(dataset, model_type, embedding_type)
                        
                        model.embedder = master_embedder 

                        start_train = datetime.now()
                        model.train_on_vectors(X_train_vec, y_train)
                        train_dur = (datetime.now() - start_train).total_seconds()

                        metrics = model.evaluate_on_vectors(X_test_vec, y_test)
                        model.save()

                        result_entry = {
                            "dataset": dataset,
                            "model": model_type,
                            "embedding": embedding_type,
                            "accuracy": round(metrics.get('accuracy', 0), 4),
                            "f1_score": round(metrics.get('f1_score', 0), 4),
                            "precision": round(metrics.get('precision', 0), 4),
                            "recall": round(metrics.get('recall', 0), 4),
                            "train_time_sec": round(train_dur, 2),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        dataset_results.append(result_entry)
                        print(f"DONE (Acc: {result_entry['accuracy']:.2f})")

                    except Exception as e:
                        print(f"FAILED: {e}")
                        traceback.print_exc()

            except Exception as e:
                print(f"[Error] Skipping embedding {embedding_type}: {e}")

        if dataset_results:
            df_results = pd.DataFrame(dataset_results)
            file_exists = os.path.isfile(RESULTS_FILE)
            df_results.to_csv(RESULTS_FILE, mode='a', header=not file_exists, index=False)
            print(f"\n[SUMMARY] Saved {len(dataset_results)} results to {RESULTS_FILE}")

    print(f"\n=== BENCHMARK COMPLETE: {datetime.now()} ===")

if __name__ == "__main__":
    run_benchmark()