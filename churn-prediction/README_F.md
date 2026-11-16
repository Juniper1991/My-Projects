# churn-prediction-app
Customer Churn Prediction — Streamlit App

pip install uv

uv venv --python 3.12 .venv

source .venv/bin/activate

uv pip install --only-binary=:all: -r requirements.txt

python -m streamlit run app.py
