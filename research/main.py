import os
import pandas as pd
from datetime import datetime
import traceback
from research.factory import ModelFactory

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
    """
    Loads data from specific file: data/processed/{dataset_name}/{split_type}.csv
    """
    file_name = f"{split_type}.csv"
    file_path = os.path.join(BASE_DATA_DIR, dataset_name, file_name)

    try:
        if not os.path.exists(file_path):
            print(f"! File does not exist: {file_path}")
            return [], []

        df = pd.read_csv(file_path)
        
        if 'text' not in df.columns or 'label' not in df.columns:
            print(f"! File {file_name} is missing columns: 'text' and 'label'")
            return [], []

        df = df.dropna(subset=['text', 'label'])
        
        texts = df['text'].astype(str).tolist()
        labels = df['label'].tolist()
        
        print(f"Loaded {len(texts)} rows from {dataset_name}/{file_name}")
        return texts, labels
        
    except Exception as e:
        print(f"! Error loading {file_path}: {e}")
        return [], []

def run_benchmark():
    """
    Main loop to iterate over datasets, models, and embeddings to record results.
    """
    print(f"STARTING RESEARCH BENCHMARK: {datetime.now()}\n")

    for dataset in DATASETS:
        dataset_results = []
        RESULTS_FILE = f"results_{dataset}.csv"
        
        print(f">> Processing Dataset: {dataset}")
        
        X_train, y_train = load_data(dataset, 'train')
        X_test, y_test = load_data(dataset, 'test')
        
        if not X_train or not X_test:
            print(f"[SKIP] Missing data for {dataset}")
            continue

        for model_type in ALL_MODELS_TO_TEST:
            try:
                valid_embeddings = ModelFactory.get_valid_embeddings_for_model(model_type)
            except: 
                valid_embeddings = []

            embeddings_to_run = set(ALL_EMBEDDINGS_TO_TEST).intersection(valid_embeddings)
            
            if not embeddings_to_run: 
                continue

            for embedding in embeddings_to_run:
                config_name = f"[{dataset}] Model: {model_type} | Embedding: {embedding}"
                print(f"[START] {config_name}")
                
                try:
                    model = ModelFactory.get_model(dataset, model_type, embedding)

                    start_time = datetime.now()
                    model.train(X_train, y_train)
                    duration = (datetime.now() - start_time).total_seconds()

                    metrics = model.evaluate(X_test, y_test)
                    model.save()
                    
                    result_entry = {
                        "dataset": dataset,
                        "model": model_type,
                        "embedding": embedding,
                        "accuracy": round(metrics.get('accuracy', 0), 4),
                        "precision": round(metrics.get('precision', 0), 4),
                        "recall": round(metrics.get('recall', 0), 4),
                        "f1_score": round(metrics.get('f1_score', 0), 4),
                        "train_time_sec": round(duration, 2),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    dataset_results.append(result_entry)
                    print(f"[OK] Acc: {metrics.get('accuracy', 0):.4f} | F1: {metrics.get('f1_score', 0):.4f}")
                    
                except Exception as e:
                    print(f"[ERROR] {e}")
                    traceback.print_exc()

        if dataset_results:
            df_results = pd.DataFrame(dataset_results)
            file_exists = os.path.isfile(RESULTS_FILE)
            df_results.to_csv(RESULTS_FILE, mode='a', header=not file_exists, index=False)
            print(f"Results for {dataset} saved in: {RESULTS_FILE}\n")
        else:
            print(f"No results found for {dataset}\n")

if __name__ == "__main__":
    run_benchmark()