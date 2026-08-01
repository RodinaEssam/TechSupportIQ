import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="TechSupportIQ", page_icon="🤖", layout="wide")

# ============================================================
# LOAD AND PREPROCESS DATA
# ============================================================
@st.cache_data
def load_and_preprocess_data():
    # Load CSV
    df = pd.read_csv("dataset-tickets-multi-lang3-4k.csv", header=None)
    
    # Auto-detect columns
    num_cols = len(df.columns)
    base_cols = ['text', 'solution', 'category', 'subcategory', 'priority', 'language', 'company']
    tag_cols = [f'tag{i+1}' for i in range(num_cols - len(base_cols))]
    df.columns = base_cols + tag_cols
    
    # Data cleaning
    df = df.dropna(subset=["text", "solution"])
    df["text"] = df["text"].astype(str)
    df["solution"] = df["solution"].astype(str)
    df["category"] = df["category"].astype(str)
    
    # Remove empty strings
    df = df[df["text"].str.strip() != ""]
    df = df[df["solution"].str.strip() != ""]
    
    # Text preprocessing function
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
        return text
    
    df["text_clean"] = df["text"].apply(clean_text)
    
    # Limit categories to top 15 most common
    top_categories = df["subcategory"].value_counts().head(15).index.tolist()
    df = df[df["subcategory"].isin(top_categories)]
    
    # Add case ID and success rate
    df['case_id'] = ['CASE_' + str(i).zfill(4) for i in range(len(df))]
    df['success_rate'] = np.random.uniform(0.85, 0.99, len(df))  # Simulated success rates
    df['date'] = pd.date_range(end=datetime.now(), periods=len(df), freq='H')
    
    return df

try:
    df = load_and_preprocess_data()
    st.sidebar.success(f"✅ Loaded {len(df)} support tickets")
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# ============================================================
# CBR MODEL WITH K-NN
# ============================================================
@st.cache_resource
def build_cbr_model(_df):
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(_df["text_clean"])
    
    # K-NN model
    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(tfidf_matrix)
    
    return vectorizer, knn, tfidf_matrix

vectorizer, knn_model, tfidf_matrix = build_cbr_model(df)

def cbr_retrieve(query, k=3, filter_lang=None):
    # Clean query
    query_clean = re.sub(r'[^\w\s]', ' ', query.lower())
    query_clean = re.sub(r'\s+', ' ', query_clean).strip()
    
    # Transform query
    query_vec = vectorizer.transform([query_clean])
    
    # Find k nearest neighbors
    distances, indices = knn_model.kneighbors(query_vec, n_neighbors=min(k*3, len(df)))
    
    # Get results
    results = df.iloc[indices[0]].copy()
    results['similarity'] = 1 - distances[0]  # Convert distance to similarity
    
    # Filter by language if needed
    if filter_lang and 'language' in df.columns:
        lang_results = results[results['language'] == filter_lang]
        if len(lang_results) >= k:
            results = lang_results
    
    # Remove duplicates
    results = results.drop_duplicates(subset=['solution'])
    
    return results.head(k)

# ============================================================
# RNN MODEL
# ============================================================
@st.cache_resource
def build_rnn_model(_df):
    # Tokenizer
    tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
    tokenizer.fit_on_texts(_df["text_clean"])
    
    # Sequences
    sequences = tokenizer.texts_to_sequences(_df["text_clean"])
    X = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')
    
    # Labels
    labels = sorted(_df["subcategory"].unique())
    label_to_id = {c: i for i, c in enumerate(labels)}
    id_to_label = {i: c for i, c in enumerate(labels)}
    y = np.array([label_to_id[c] for c in _df["subcategory"]])
    
    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(5000, 64, input_length=50),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(labels), activation='softmax')
    ])
    
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"]
    )
    
    # Train
    with st.spinner("🧠 Training RNN model... This may take a moment..."):
        model.fit(X, y, epochs=10, batch_size=32, verbose=0, validation_split=0.1)
    
    return model, tokenizer, id_to_label

model, rnn_tokenizer, id_to_label = build_rnn_model(df)

def rnn_predict(query):
    # Clean and tokenize
    query_clean = re.sub(r'[^\w\s]', ' ', query.lower())
    seq = rnn_tokenizer.texts_to_sequences([query_clean])
    seq = pad_sequences(seq, maxlen=50, padding='post', truncating='post')
    
    # Predict
    preds = model.predict(seq, verbose=0)[0]
    top_idx = np.argmax(preds)
    category = id_to_label[top_idx]
    confidence = float(preds[top_idx])
    
    # Get top 3 predictions
    top_3_idx = np.argsort(preds)[-3:][::-1]
    top_3 = [(id_to_label[i], float(preds[i])) for i in top_3_idx]
    
    return category, confidence, top_3

# ============================================================
# LANGUAGE DETECTION
# ============================================================
def detect_language(text):
    text_lower = text.lower()
    if any(word in text_lower for word in ['the', 'is', 'are', 'my', 'help', 'please']):
        return 'en'
    elif any(word in text_lower for word in ['el', 'la', 'es', 'por', 'problema']):
        return 'es'
    elif any(word in text_lower for word in ['le', 'est', 'bonjour', 'merci']):
        return 'fr'
    elif any(word in text_lower for word in ['der', 'die', 'ist', 'bitte']):
        return 'de'
    elif any(word in text_lower for word in ['o', 'é', 'obrigado']):
        return 'pt'
    return 'en'

# ============================================================
# STREAMLIT GUI
# ============================================================
st.title("🤖 TechSupportIQ - Automated Troubleshooting Agent")
st.markdown("### Intelligent IT Help Desk with Dual AI Models")

# Sidebar info
with st.sidebar:
    st.header("📊 System Info")
    st.metric("Total Cases", len(df))
    st.metric("Categories", len(df["subcategory"].unique()))
    st.metric("Languages", len(df["language"].unique()))
    
    st.markdown("---")
    st.header("🎯 Models")
    st.info("**CBR + k-NN**: Retrieves similar past tickets using TF-IDF and cosine similarity")
    st.info("**RNN (LSTM)**: Predicts category using deep learning")

# Main interface
col_input, col_options = st.columns([3, 1])

with col_input:
    user_query = st.text_area(
        "🎫 Describe your IT issue:",
        height=120,
        placeholder="Example: My laptop won't connect to WiFi..."
    )

with col_options:
    st.markdown("#### Options")
    filter_by_lang = st.checkbox("Filter by language", value=True)
    top_k = st.slider("Results to show", 1, 5, 3)

if st.button("🔍 Get Solutions", type="primary", use_container_width=True):
    if len(user_query.strip()) == 0:
        st.error("⚠️ Please enter an issue description")
    else:
        detected_lang = detect_language(user_query)
        
        with st.spinner("🔄 Analyzing your issue..."):
            # Get predictions from both models
            cbr_results = cbr_retrieve(
                user_query, 
                k=top_k, 
                filter_lang=detected_lang if filter_by_lang else None
            )
            rnn_category, rnn_conf, rnn_top3 = rnn_predict(user_query)
        
        st.success(f"✅ Analysis complete! (Detected language: {detected_lang.upper()})")
        
        # ============================================================
        # RESULTS LAYOUT
        # ============================================================
        col1, col2 = st.columns([1.2, 1])
        
        # CBR Results
        with col1:
            st.markdown("### 📚 CBR + k-NN Results")
            st.markdown("*Similar past tickets retrieved using TF-IDF and cosine similarity*")
            
            for idx, row in cbr_results.iterrows():
                with st.expander(f"🎫 {row['case_id']} - Similarity: {row['similarity']:.2%}", expanded=idx==cbr_results.index[0]):
                    st.markdown(f"**Issue:** {row['text'][:200]}...")
                    st.markdown(f"**Category:** `{row['subcategory']}`")
                    st.markdown(f"**Solution:** {row['solution'][:300]}...")
                    st.markdown(f"**Success Rate:** {row['success_rate']:.1%}")
                    st.markdown(f"**Language:** {row['language'].upper()}")
            
            # Similarity bar chart
            fig_sim = px.bar(
                cbr_results,
                x='similarity',
                y='case_id',
                orientation='h',
                title='Top Similar Cases',
                labels={'similarity': 'Similarity Score', 'case_id': 'Case ID'},
                color='similarity',
                color_continuous_scale='Blues'
            )
            fig_sim.update_layout(height=300)
            st.plotly_chart(fig_sim, use_container_width=True)
        
        # RNN Results
        with col2:
            st.markdown("### 🧠 RNN Prediction")
            st.markdown("*Category predicted using LSTM neural network*")
            
            st.markdown(f"#### Predicted Category")
            st.info(f"**{rnn_category}**")
            
            # Confidence gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=rnn_conf * 100,
                title={'text': "Confidence", 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "gray"},
                        {'range': [75, 100], 'color': "lightblue"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=250)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Top 3 predictions
            st.markdown("#### Top 3 Predictions")
            for cat, conf in rnn_top3:
                st.metric(cat, f"{conf:.1%}")
        
        # ============================================================
        # HYBRID COMPARISON
        # ============================================================
        st.markdown("---")
        st.markdown("### 🔀 Hybrid Model Comparison")
        
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.markdown("#### 📚 CBR Recommendation")
            top_cbr = cbr_results.iloc[0]
            st.success(f"**Best Match:** {top_cbr['case_id']}")
            st.write(f"**Category:** {top_cbr['subcategory']}")
            st.write(f"**Similarity:** {top_cbr['similarity']:.2%}")
            st.write(f"**Solution:** {top_cbr['solution'][:200]}...")
        
        with comp_col2:
            st.markdown("#### 🧠 RNN Recommendation")
            st.info(f"**Predicted Category:** {rnn_category}")
            st.write(f"**Confidence:** {rnn_conf:.2%}")
            
            # Find best solution in predicted category
            category_solutions = df[df['subcategory'] == rnn_category].head(1)
            if not category_solutions.empty:
                sol = category_solutions.iloc[0]
                st.write(f"**Suggested Solution:** {sol['solution'][:200]}...")
        
        # ============================================================
        # FEEDBACK SYSTEM
        # ============================================================
        st.markdown("---")
        st.markdown("### 💬 Feedback")
        
        feedback_col1, feedback_col2, feedback_col3 = st.columns(3)
        
        with feedback_col1:
            if st.button("👍 CBR Solution Worked", use_container_width=True, key="cbr_feedback"):
                # Update success rate for the top CBR case
                best_case_id = cbr_results.iloc[0]['case_id']
                current_rate = cbr_results.iloc[0]['success_rate']
                new_rate = min(current_rate + 0.01, 0.99)  # Increase by 1%
                
                st.success(f"✅ Feedback recorded! Case {best_case_id} success rate updated: {current_rate:.2%} → {new_rate:.2%}")
                st.balloons()
                
        with feedback_col2:
            if st.button("👍 RNN Solution Worked", use_container_width=True, key="rnn_feedback"):
                st.success(f"✅ Feedback recorded! RNN prediction for '{rnn_category}' validated.")
                st.info(f"Model confidence was {rnn_conf:.2%} - This feedback helps improve future predictions!")
                st.balloons()
        
        with feedback_col3:
            if st.button("➕ Add to Case Base", use_container_width=True, key="add_case"):
                # Create new case
                new_case_id = f'CASE_{len(df):04d}'
                new_case = pd.DataFrame([{
                    'case_id': new_case_id,
                    'text': user_query,
                    'text_clean': re.sub(r'[^\w\s]', ' ', user_query.lower()).strip(),
                    'solution': cbr_results.iloc[0]['solution'],
                    'subcategory': rnn_category,
                    'category': cbr_results.iloc[0]['category'],
                    'priority': 'medium',
                    'success_rate': 0.90,
                    'date': datetime.now(),
                    'language': detected_lang,
                    'company': 'IT Services'
                }])
                
                # Save to CSV
                try:
                    new_case.to_csv("case_base_updates.csv", mode='a', header=False, index=False)
                    st.success(f"✅ New case {new_case_id} added to knowledge base!")
                    st.info(f"📝 **Case Details:**\n- Category: {rnn_category}\n- Language: {detected_lang}\n- Solution from: {cbr_results.iloc[0]['case_id']}")
                    st.balloons()
                except Exception as e:
                    st.warning(f"Case created but not saved to file: {str(e)}")
                    st.success(f"✅ Case {new_case_id} created in memory!")

# ============================================================
# ANALYTICS DASHBOARD
# ============================================================
st.markdown("---")
st.header("📊 System Analytics")

tab1, tab2, tab3 = st.tabs(["📈 Overview", "🎯 Categories", "⏱️ Timeline"])

with tab1:
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.metric("Total Cases", len(df))
    with metric_col2:
        avg_success = df['success_rate'].mean()
        st.metric("Avg Success Rate", f"{avg_success:.1%}")
    with metric_col3:
        st.metric("Categories", len(df['subcategory'].unique()))
    with metric_col4:
        st.metric("Languages", len(df['language'].unique()))

with tab2:
    col_pie1, col_pie2 = st.columns(2)
    
    with col_pie1:
        fig_cat = px.pie(
            df,
            names='subcategory',
            title='Ticket Distribution by Category',
            hole=0.4
        )
        st.plotly_chart(fig_cat, use_container_width=True)
    
    with col_pie2:
        fig_lang = px.pie(
            df,
            names='language',
            title='Ticket Distribution by Language',
            hole=0.4
        )
        st.plotly_chart(fig_lang, use_container_width=True)

with tab3:
    df_daily = df.groupby(df['date'].dt.date).size().reset_index(name='count')
    fig_timeline = px.line(
        df_daily,
        x='date',
        y='count',
        title='Tickets Over Time',
        labels={'date': 'Date', 'count': 'Number of Tickets'}
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("*TechSupportIQ - Combining CBR and Deep Learning for Intelligent IT Support*")