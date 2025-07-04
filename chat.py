import streamlit as st
import pandas as pd
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import re

@st.cache_data
def get_top_tags(n=4):
    tag_df = pd.read_csv("tag_counts.csv")
    top_tags = tag_df.sort_values("count", ascending=False)["tags_cleaned"].head(n).tolist()
    return set(top_tags)

TOP_4_TAGS = get_top_tags()

# ─── Configuration ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = "AIzaSyDjmJZB7jp92jiKQHHu9gn86gpjLD_30NQ"  
genai.configure(api_key=GEMINI_API_KEY)
gemini = genai.GenerativeModel("gemini-2.5-flash")

# Use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

# ─── Data Processing Functions ──────────────────────────────────────────────────
def clean_and_split_tags(tag_input):
    """Clean and split tags from various input formats"""
    if pd.isna(tag_input) or tag_input == "" or tag_input == "nan":
        return []
    
    # Convert to string
    tag_string = str(tag_input)
    
    # Handle list-like strings (e.g., "['tag1', 'tag2']")
    if tag_string.startswith('[') and tag_string.endswith(']'):
        try:
            # Remove brackets and quotes, then split
            tag_string = tag_string.strip('[]')
            # Split by comma and clean each tag
            tags = []
            for tag in tag_string.split(','):
                # Remove quotes and whitespace
                clean_tag = tag.strip().strip("'").strip('"').strip().lower()
                if clean_tag:
                    tags.append(clean_tag)
            return tags
        except:
            # Fallback to regular comma splitting
            pass
    
    # Handle regular comma-separated strings
    tags = [tag.strip().lower() for tag in tag_string.split(',')]
    return [tag for tag in tags if tag and tag != 'nan']  # Remove empty tags

def calculate_tag_metrics(true_tags_list, pred_tags_list):
    """Calculate precision, recall, and F1 for tags"""
    if not true_tags_list and not pred_tags_list:
        return 1.0, 1.0, 1.0  # Perfect match for empty sets
    
    true_tags_set = set(true_tags_list)
    pred_tags_set = set(pred_tags_list)
    
    if not pred_tags_set:
        return 0.0, 0.0, 0.0
    
    intersection = len(true_tags_set.intersection(pred_tags_set))
    
    precision = intersection / len(pred_tags_set) if pred_tags_set else 0.0
    recall = intersection / len(true_tags_set) if true_tags_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1

# ─── Data Cleaning Function ────────────────────────────────────────────────────
def clean_dataframe(df):
    """Clean and standardize DataFrame column types"""
    df_clean = df.copy()
    
    # Convert all columns to string type to avoid mixed type issues
    for col in df_clean.columns:
        df_clean[col] = df_clean[col].astype(str)
    
    # Replace 'nan' strings with empty strings
    df_clean = df_clean.replace('nan', '')
    
    return df_clean

# ─── Data Load + Split + FAISS ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_split_data(test_size=0.2, random_state=42):
    df = pd.read_csv("vehicle_specs.csv")
    df = clean_dataframe(df)
    
    # Split into train and test
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    # Create embeddings only from training data (excluding tags and additional_info)
    feature_columns = [col for col in train_df.columns if col not in ['tags', 'additional_info']]
    train_texts = [" | ".join(f"{k}: {v}" for k, v in row[feature_columns].items()) 
                   for _, row in train_df.iterrows()]
    
    embeddings = model.encode(train_texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    
    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    return train_df, test_df, train_texts, embeddings, index, feature_columns

train_df, test_df, train_texts, embeddings, faiss_index, feature_columns = load_and_split_data()

# ─── MMR Function ───────────────────────────────────────────────────────────────
def rerank_mmr(embeddings_subset, query_embedding, k=5, diversity=0.7):
    selected, candidates = [], list(range(len(embeddings_subset)))
    query_tensor = torch.tensor(query_embedding).unsqueeze(0)
    embeddings_tensor = torch.tensor(embeddings_subset)
    sim = torch.nn.functional.cosine_similarity(query_tensor, embeddings_tensor, dim=1).numpy()
    
    for _ in range(min(k, len(candidates))):
        if not selected:
            idx = int(np.argmax(sim))
        else:
            mmr_scores = []
            for i in candidates:
                rel = sim[i]
                div = max(torch.nn.functional.cosine_similarity(
                    embeddings_tensor[i].unsqueeze(0),
                    embeddings_tensor[selected]
                ).numpy())
                mmr = diversity * rel - (1 - diversity) * div
                mmr_scores.append((i, mmr))
            idx = max(mmr_scores, key=lambda x: x[1])[0]
        selected.append(idx)
        candidates.remove(idx)
    return selected

# ─── Prediction Function ───────────────────────────────────────────────────────
def predict_tags_and_info(user_input, top_k=5, diversity=0.7):
    """Predict tags and additional info for given input specifications"""
    query_vec = model.encode([user_input], normalize_embeddings=True)[0].astype("float32")
    _, faiss_ids = faiss_index.search(np.array([query_vec]), top_k)
    subset_indices = faiss_ids[0].tolist()
    subset_embeddings = embeddings[subset_indices]
    reranked = rerank_mmr(subset_embeddings, query_vec, k=top_k, diversity=diversity)
    
    # Get context from training data (including tags and additional_info for examples)
    context_rows = train_df.iloc[subset_indices]
    context_texts = []
    for i in reranked:
        row = context_rows.iloc[i]
        context_text = " | ".join(f"{k}: {v}" for k, v in row.items())
        context_texts.append(context_text)
    
    context = "\n".join(f"Example {i+1}: {t}" for i, t in enumerate(context_texts))
    
    prompt = f"""You are an expert in vehicle specifications analysis.

Based on these training examples:
{context}

For the following vehicle specifications:
{user_input}

Please predict:
1. Tags: (provide as comma-separated list of relevant tags/categories)
2. Additional Info: (provide brief summary or relevant information)

Format your response exactly as:
Tags: [your comma-separated tags here]
Additional Info: [your summary here]"""
    
    response = gemini.generate_content(prompt)
    return response.text, context

# ─── Evaluation Function ───────────────────────────────────────────────────────
def evaluate_model(test_sample_size=10, top_k=5, diversity=0.7):
    """Evaluate model on test set"""
    # Sample from test set for evaluation
    test_sample = test_df.sample(n=min(test_sample_size, len(test_df)), random_state=42)
    
    results = []
    tag_metrics = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, (_, row) in enumerate(test_sample.iterrows()):
        status_text.text(f"Evaluating sample {idx+1}/{len(test_sample)}")
        test_input = " | ".join(f"{k}: {v}" for k, v in row[feature_columns].items())
        true_tags = clean_and_split_tags(row['tags'])
        true_info = str(row['additional_info']) if pd.notna(row['additional_info']) else ""

        try:
            prediction, context = predict_tags_and_info(test_input, top_k, diversity)
            pred_tags = []
            pred_info = ""
            prediction_clean = prediction.strip()
            lines = prediction_clean.split('\n')
            for line in lines:
                line = line.strip()
                if line.lower().startswith('tags:'):
                    tags_text = line[5:].strip('[](){}')
                    pred_tags = clean_and_split_tags(tags_text)
                elif line.lower().startswith('additional info:'):
                    pred_info = line[16:].strip()
            # Remove top 4 tags from both
            true_tags = [t for t in true_tags if t not in TOP_4_TAGS]
            pred_tags = [t for t in pred_tags if t not in TOP_4_TAGS]
            # Calculate metrics
            precision, recall, f1 = calculate_tag_metrics(true_tags, pred_tags)
            tag_metrics.append((precision, recall, f1))
            results.append({
                'input': test_input,
                'true_tags': true_tags,
                'pred_tags': pred_tags,
                'true_info': true_info,
                'pred_info': pred_info,
                'tag_precision': precision,
                'tag_recall': recall,
                'tag_f1': f1
            })
        except Exception as e:
            st.warning(f"Error processing sample {idx+1}: {e}")
            continue
        progress_bar.progress((idx + 1) / len(test_sample))
    
    status_text.empty()
    progress_bar.empty()
    
    return results, tag_metrics

# ─── UI ─────────────────────────────────────────────────────────────────────────
st.title(":hammer_and_wrench: Vehicle Tag & Info Predictor with Evaluation")
st.markdown("Train/Test split implementation with precision and recall calculation")

# Display dataset info
st.sidebar.markdown("### Dataset Info")
st.sidebar.write(f"Training samples: {len(train_df)}")
st.sidebar.write(f"Test samples: {len(test_df)}")
st.sidebar.write(f"Feature columns: {len(feature_columns)}")

# Tab layout
tab1, tab2, tab3 = st.tabs(["Manual Prediction", "Model Evaluation", "Dataset Preview"])

with tab1:
    st.header("Manual Prediction")
    with st.form("input_form"):
        user_input = st.text_area(":wrench: Enter Specs (Format: key: value | key: value ...)")
        top_k = st.slider("Top similar specs to retrieve", 1, 10, 5)
        diversity = st.slider("MMR Diversity (0 = focused, 1 = diverse)", 0.0, 1.0, 0.7, 0.05)
        submitted = st.form_submit_button(":mag: Predict Tags & Info")

    if submitted and user_input.strip():
        try:
            start = time.time()
            prediction, context = predict_tags_and_info(user_input, top_k, diversity)

            # Parse tags from prediction
            pred_tags = []
            for line in prediction.strip().split('\n'):
                if line.lower().startswith('tags:'):
                    tags_text = line[5:].strip().strip('[](){}')
                    pred_tags = clean_and_split_tags(tags_text)
            # Remove top 4 tags
            pred_tags = [t for t in pred_tags if t not in TOP_4_TAGS]

            st.success(":white_check_mark: Predicted Output (Top 4 tags removed)")
            st.markdown(f"**Tags:** {', '.join(pred_tags)}")
            st.markdown(prediction)

            with st.expander(":brain: Context Used from Training Data"):
                st.text(context)

            st.markdown(f":zap: Processed in `{round(time.time() - start, 2)}` seconds.")

            # Optional: Allow user to enter ground truth tags for quick metrics
            ground_truth_tags = st.text_input("Enter ground truth tags (comma-separated):")
            if ground_truth_tags:
                gt_tags = set([t.strip() for t in ground_truth_tags.split(",") if t.strip() and t.strip() not in TOP_4_TAGS])
                pred_tags_set = set(pred_tags)
                true_positives = len(gt_tags & pred_tags_set)
                precision = true_positives / len(pred_tags_set) if pred_tags_set else 0
                recall = true_positives / len(gt_tags) if gt_tags else 0
                st.info(f"Precision: {precision:.2f}, Recall: {recall:.2f}")

                # Tag-wise accuracy
                all_tags = gt_tags | pred_tags_set
                tag_accuracy = {tag: int(tag in gt_tags and tag in pred_tags_set) for tag in all_tags}
                st.write("Tag-wise accuracy (1=correct, 0=incorrect):")
                st.json(tag_accuracy)

        except Exception as e:
            st.error(f":x: Error during prediction: {e}")

with tab2:
    st.header("Model Evaluation")
    
    with st.form("eval_form"):
        eval_sample_size = st.slider("Number of test samples to evaluate", 5, 50, 10)
        eval_top_k = st.slider("Top K for evaluation", 1, 10, 5)
        eval_diversity = st.slider("MMR Diversity for evaluation", 0.0, 1.0, 0.7, 0.05)
        eval_submitted = st.form_submit_button(":chart_with_upwards_trend: Run Evaluation")
    
    if eval_submitted:
        st.info("Running evaluation... This may take a few minutes.")
        
        try:
            results, tag_metrics = evaluate_model(eval_sample_size, eval_top_k, eval_diversity)
            
            if tag_metrics:
                # Calculate overall metrics
                avg_precision = np.mean([m[0] for m in tag_metrics])
                avg_recall = np.mean([m[1] for m in tag_metrics])
                avg_f1 = np.mean([m[2] for m in tag_metrics])
                
                # Display overall metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Precision", f"{avg_precision:.3f}")
                with col2:
                    st.metric("Average Recall", f"{avg_recall:.3f}")
                with col3:
                    st.metric("Average F1-Score", f"{avg_f1:.3f}")
                
                # Detailed results
                st.subheader("Detailed Results")
                results_df = pd.DataFrame(results)
                # Convert list columns to strings for display
                display_results = results_df.copy()
                display_results['true_tags'] = display_results['true_tags'].apply(lambda x: ', '.join(x) if x else '')
                display_results['pred_tags'] = display_results['pred_tags'].apply(lambda x: ', '.join(x) if x else '')
                
                # Select columns for display
                display_cols = ['true_tags', 'pred_tags', 'tag_precision', 'tag_recall', 'tag_f1']
                st.dataframe(display_results[display_cols], use_container_width=True)
                
                # Show some examples
                st.subheader("Sample Predictions")
                for i, result in enumerate(results[:3]):  # Show first 3 examples
                    with st.expander(f"Example {i+1} - P:{result['tag_precision']:.3f} R:{result['tag_recall']:.3f} F1:{result['tag_f1']:.3f}"):
                        st.write("**Input:**", result['input'][:200] + "..." if len(result['input']) > 200 else result['input'])
                        
                        # Display tags comparison
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**True Tags:**")
                            st.code(str(result['true_tags']))
                        with col2:
                            st.write("**Predicted Tags:**")
                            st.code(str(result['pred_tags']))
                        
                        # Tag comparison details
                        true_set = set(result['true_tags'])
                        pred_set = set(result['pred_tags'])
                        intersection = true_set.intersection(pred_set)
                        
                        if intersection:
                            st.success(f"✅ Matched tags: {list(intersection)}")
                        if true_set - pred_set:
                            st.error(f"❌ Missed tags: {list(true_set - pred_set)}")
                        if pred_set - true_set:
                            st.warning(f"⚠️ Extra tags: {list(pred_set - true_set)}")
                        
                        st.write("**True Info:**", result['true_info'][:100] + "..." if len(result['true_info']) > 100 else result['true_info'])
                        st.write("**Predicted Info:**", result['pred_info'][:100] + "..." if len(result['pred_info']) > 100 else result['pred_info'])
            
        except Exception as e:
            st.error(f"Error during evaluation: {e}")

with tab3:
    st.header("Dataset Preview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Data Sample")
        # Display with proper formatting
        display_train = train_df.head().copy()
        st.dataframe(display_train, use_container_width=True)
    
    with col2:
        st.subheader("Test Data Sample")
        # Display with proper formatting
        display_test = test_df.head().copy()
        st.dataframe(display_test, use_container_width=True)
    
    st.subheader("Feature Columns Used for Prediction")
    st.write(feature_columns)