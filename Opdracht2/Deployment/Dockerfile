FROM python:3.11.5

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app files (and model file)
COPY app.py .
COPY regression_model.pkl .
COPY regression_scaler.pkl .

# Render sets $PORT for web services

CMD ["sh","-c","streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT"]