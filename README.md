# Fake News Detector & Research Framework

A comprehensive machine learning project that detects fake news using various NLP techniques. This repository contains both a **Research Framework** for benchmarking models (SVM, Logistic Regression, etc.) on multiple datasets (ISOT, LIAR, WELFake) and a **Chrome Extension** powered by a FastAPI backend to detect fake news in real-time.

---

## 🚀 Features

* **Modular Architecture:** Uses Factory Pattern to easily swap models and embeddings.
* **Multiple Embeddings:** Support for TF-IDF, Word2Vec, and modularity for future Transformers (BERT).
* **Smart Routing:** The API analyzes text length and dynamically selects the best model:
    * *Short text* (< 30 words) → Uses model trained on **LIAR** (claims/statements).
    * *Long text* (> 100 words) → Uses model trained on **ISOT/WELFake** (full articles).
* **Chrome Extension:** A user-friendly browser popup to analyze selected text on any webpage.
* **Research Pipeline:** Automated training and evaluation scripts producing CSV reports.

---

## Project Structure

```text
/project_root
├── backend/               # FastAPI Backend (The brain of the extension)
│   └── api.py             # Router logic and model serving
├── extension/             # Chrome Extension (Frontend)
│   ├── manifest.json
│   ├── popup.html
│   ├── popup.js
│   └── styles.css
├── research/              # Machine Learning Logic
│   ├── data/              # Dataset directory (gitignored)
│   ├── embeddings/        # Embeddings implementations
│   ├── models/            # Models wrappers
│   ├── base.py            # Abstract base classes
│   ├── main.py            # Script to run research benchmarks
│   └── factory.py         # Model Factory pattern
├── saved_models/          # Serialized models (.joblib, .pkl)
└── requirements.txt       # Python dependencies
```

---

### Installation & Setup
1. Clone the Repository
```bash
git clone [https://github.com/yourusername/fake-news-detector.git](https://github.com/yourusername/fake-news-detector.git)
cd fake-news-detector
```
2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Prepare Data
* Since datasets are heavy, they are not included in the repo.
* Download ISOT, LIAR, and WELFake datasets.
* Place .csv files in the corresponding folders (research/data/rocessed/...)

---


### How to Run Research (Training)
To train models and generate benchmark results: 
1. Open main.py.
2. Configure the MODELS_TO_TRAIN and DATASETS lists.
3. Run the script:
```bash
python main.py
```

This will: 
* Train the selected models.
* Evaluate them on test sets. Save trained models to saved_models/.
* Save performance metrics (Accuracy, F1-Score) to results_*.csv.

---

### How to Run the Chrome Extension
1. Start the Backend API

The backend serves the trained models. It must be running for the extension to work.
```bash
python backend/api.py
```

You should see output indicating that models are being loaded into memory.

2. Install Extension in Chrome
   * Open Chrome and go to chrome://extensions/.
   * Toggle Developer mode (top right).
   * Click Load unpacked.
   * Select the extension/ folder from this project.
  
3. Use it!
   * Go to any news website.
   *  Select text (highlight a paragraph or headline).
   *  Click the Fake News Detector icon in your browser toolbar.
   *  Click ANALYZE SELECTION.

---
