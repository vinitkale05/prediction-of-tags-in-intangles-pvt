# prediction-of-tags-in-intangles-pvt
Project where tags can be recommended using ai in intangles lab pvt
# 🚗 Vehicle Tag & Info Predictor with Model Evaluation

This Streamlit app predicts vehicle tags and additional information from user-entered vehicle specifications, using a combination of semantic search (SentenceTransformer + FAISS), Gemini LLM for generation, and robust model evaluation. It allows manual predictions as well as automated evaluation over a test set with detailed precision, recall, and F1-score analysis.

---

## Features

- **Manual Tag Prediction:**  
  Enter vehicle specs in a key-value format, retrieve similar training examples, and get AI-predicted tags and additional info.
- **Model Evaluation:**  
  Evaluate the model's tag prediction performance on a test set with interactive controls for sample size, top-K retrieval, and MMR diversity. Displays precision, recall, F1-score, and detailed results.
- **Dataset Preview:**  
  Inspect samples from the training and test sets, and review feature columns used for embedding and search.
- **Interactive Controls:**  
  Adjust retrieval diversity, number of neighbors, and evaluation parameters via sliders.
- **Contextual LLM Prompting:**  
  Uses Gemini API to generate predictions, provided with context from similar training examples.
- **Custom Tag Cleaning & Metrics:**  
  Handles various tag formats, removes top-4 most common tags for fair evaluation, and computes per-sample metrics.

---

## Installation & Setup

1. **Clone the Repository**

    ```bash
    git clone <your-repo-url>
    cd <your-repo-folder>
    ```

2. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

    *Main dependencies:*
    - `streamlit`
    - `pandas`, `numpy`
    - `torch`
    - `faiss-cpu` or `faiss-gpu`
    - `sentence-transformers`
    - `google-generativeai`
    - `scikit-learn`

3. **Prepare Data**

    - Place your cleaned vehicle data as `vehicle_specs.csv` in the project root.  
      **Columns required:**  
      - Feature columns (any vehicle specs, e.g., make, model, year, engine, etc.)
      - `tags` (comma-separated or list-like string)
      - `additional_info` (free text, optional)

    - Prepare a `tag_counts.csv` file with columns:  
      - `tags_cleaned`
      - `count`  
      (Used to filter most common tags during evaluation.)

4. **Set Up Gemini API Key**

    - Obtain an API key for [Google Gemini](https://ai.google.dev/).
    - Replace the placeholder in `chat.py`:
      ```python
      GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
      ```

5. **Run the App**

    ```bash
    streamlit run chat.py
    ```

---

## Usage

- **Manual Prediction:**  
  - Go to the "Manual Prediction" tab.
  - Enter vehicle specifications in the format:  
    ```
    make: Toyota | model: Camry | year: 2019 | engine: 2.5L | ...
    ```
  - Adjust retrieval/top-K and diversity as needed.
  - View predicted tags and info, and optionally enter ground truth tags to see precision/recall.

- **Model Evaluation:**  
  - Go to the "Model Evaluation" tab.
  - Select number of test samples, top-K retrieval, and MMR diversity.
  - Click "Run Evaluation" to see average precision, recall, F1-score, and detailed sample results.

- **Dataset Preview:**  
  - Inspect training and test set samples.
  - Review feature columns used for model input.

---

## Project Structure

```
├── chat.py             # Main Streamlit application
├── vehicle_specs.csv   # Vehicle specs dataset (not included)
├── tag_counts.csv      # Tag frequency file (not included)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Notes

- **Data Privacy:**  
  No data is sent or logged outside your environment except for LLM inference via the Gemini API.
- **Customization:**  
  - You can adjust the feature columns, model, or tags cleaning logic as needed.
  - The top-4 tags filtering is configurable via `tag_counts.csv`.

---

## License

MIT License

---

## Acknowledgements

- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Google Gemini](https://ai.google.dev/)
- [Streamlit](https://streamlit.io/)

---
