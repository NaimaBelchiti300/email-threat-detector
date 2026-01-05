# Email Threat Detector

## Description
This project detects email threats such as **spam** and **phishing** using **Machine Learning**.  
It is designed as an intermediate-level cybersecurity mini-project for educational purposes.

Features:
- Text preprocessing (lowercasing, punctuation removal, stopwords removal, stemming)
- ML model (Naive Bayes) with evaluation metrics
- Terminal input mode
- Optional Streamlit interface for real-time testing
- Model and vectorizer saved for reuse

## Project Structure
email-threat-detector/
│
├── data/ → Dataset CSV file (spam, phishing, normal emails)
├── src/ → Source code
│ ├── main.py → Terminal / Streamlit detection
│ ├── preprocess.py → Text cleaning functions
│ └── train_model.py → Train and save ML model
├── venv/ → Virtual environment (ignored in Git)
├── requirements.txt → Python dependencies
└── README.md → Project documentation

## Installation
1. Clone the repository:
git clone https://github.com/NaimaBelchiti300/email-threat-detector

cd email-detector

2. Create a virtual environment:
python3 -m venv venv
source venv/bin/activate # Linux / Mac
venv\Scripts\activate # Windows

3. Install dependencies:
pip install --upgrade pip
pip install -r requirements.txt

## Usage

1. Train the model:
python src/train_model.py

2. Run detection:
- Choose mode: `terminal` or `streamlit` 
- terminal mode:
python src/main.py --mode terminal 
- Streamlit mode:
streamlit run src/main.py

## Dataset
- CSV format: `data/dataset.csv`  
- Columns: `text` (email content), `label` (spam/phishing/normal)

## Author
Naima Belchiti - 3ᵉ année Génie Informatique
