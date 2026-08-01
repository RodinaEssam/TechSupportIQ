# 🤖 TechSupportIQ - Hybrid AI IT Support System

TechSupportIQ is an intelligent IT help desk application that combines **Case-Based Reasoning (CBR)** with **Deep Learning (RNN/LSTM)** to provide automated troubleshooting and support ticket resolution.

The system retrieves similar historical support tickets using a Case-Based Reasoning approach while simultaneously predicting the issue category using a Recurrent Neural Network. This hybrid approach improves both recommendation accuracy and decision support.

---

## Features

- Case-Based Reasoning (CBR) using TF-IDF and k-Nearest Neighbors
- Deep Learning classification using Bidirectional LSTM
- Multi-language ticket support
- Interactive analytics dashboard
- Similarity visualization
- Confidence score for predictions
- User feedback system
- Add new cases to the knowledge base
- Built with Streamlit

---

## Dataset

This project uses the following dataset:

**dataset-tickets-multi-lang3-4k.csv**

Dataset characteristics:

- Approximately 4,000 IT support tickets
- Multiple languages
- Ticket description
- Solution
- Category
- Subcategory
- Priority
- Language
- Company
- Additional tags

---

## Technologies Used

- Python
- Streamlit
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Plotly

---

## AI Models

### 1. Case-Based Reasoning (CBR)

The CBR model retrieves similar historical tickets using:

- TF-IDF Vectorization
- Cosine Similarity
- k-Nearest Neighbors (k-NN)

This allows the system to recommend solutions that successfully solved similar issues in the past.

---

### 2. Recurrent Neural Network (RNN)

The deep learning model uses:

- Tokenizer
- Word Embedding
- Bidirectional LSTM
- Dense Layers
- Softmax Classification

The RNN predicts the most likely ticket category together with a confidence score.

---

## How It Works

1. User enters an IT issue.
2. The system detects the language.
3. The CBR model retrieves the most similar historical tickets.
4. The RNN predicts the issue category.
5. Both models provide recommendations.
6. Results are displayed with:
   - Similarity scores
   - Predicted category
   - Confidence level
   - Recommended solution
7. Users can provide feedback or add the issue to the case base.

---

## Analytics Dashboard

The dashboard includes:

- Total support cases
- Average success rate
- Ticket distribution by category
- Language distribution
- Timeline of support tickets

---

## License

This project is intended for educational and research purposes.
