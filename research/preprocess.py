import os
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import nltk
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

STOP_WORDS = set(stopwords.words('english'))

RAW_BASE = os.path.join("research", "data", "raw")
PROCESSED_BASE = os.path.join("research", "data", "processed")

def load_and_standardize(dataset, path):
    print(f"[{dataset}] Loading and initial cleaning")
    file_path = os.path.join(path, "data.csv")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return pd.DataFrame()

    df = pd.read_csv(file_path)

    df['label'] = df['label'].apply(standardize_label)
    df.dropna(subset=['label'], inplace=True)

    if dataset in ["ISOT", "WELFake"]:
        df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    elif dataset == "LIAR":
        df["text"] = df["text"].fillna("")

    df["text"] = df["text"].astype(str).str.strip()
    
    initial_len = len(df)
    df.drop_duplicates(subset=['text'], inplace=True)
    print(f"[{dataset}] Removed {initial_len - len(df)} internal duplicates.")

    return df[["text", "label"]]

def prepare_dataset(name):
    raw_path = os.path.join(RAW_BASE, name)
    out_path = os.path.join(PROCESSED_BASE, name)
    os.makedirs(out_path, exist_ok=True)

    df = load_and_standardize(name, raw_path)
    if df.empty:
        return

    train, temp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    val, test = train_test_split(temp, test_size=0.5, stratify=temp["label"], random_state=42)

    print(f"[{name}] Checking for data leakage")
    
    train_texts_set = set(train["text"])

    def remove_leakage(subset_df):
        clean_df = subset_df[~subset_df["text"].isin(train_texts_set)]
        return clean_df

    val = remove_leakage(val)
    test = remove_leakage(test)

    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(train[["text"]], train["label"])

    train_res = pd.DataFrame(X_train_res, columns=["text"])
    train_res["label"] = y_train_res

    print(f"[{name}] Final sizes: Train={len(train_res)}, Val={len(val)}, Test={len(test)}")
    
    overlap = set(train_res["text"]).intersection(set(test["text"]))
    if len(overlap) > 0:
        print(f"Detected {len(overlap)} overlapping texts!")
    else:
        print(f"[{name}] No data leakage.")

    train_res.to_csv(os.path.join(out_path, "train.csv"), index=False)
    val.to_csv(os.path.join(out_path, "val.csv"), index=False)
    test.to_csv(os.path.join(out_path, "test.csv"), index=False)

def standardize_label(label):
    if isinstance(label, (int, float)):
        return int(label)
    label_str = str(label).strip().lower()
    if label_str in ["true", "1", "real"]:
        return 1
    elif label_str in ["false", "0", "fake"]:
        return 0
    else:
        return 0 
    
if __name__ == "__main__":
    os.makedirs(PROCESSED_BASE, exist_ok=True)
    
    for dataset in ["LIAR", "ISOT", "WELFake"]:
        prepare_dataset(dataset)