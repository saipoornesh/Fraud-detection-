import os
from dotenv import load_dotenv
load_dotenv()
DATABASE_URL=os.getenv("DATABASE_URL","sqlite:///./fintech_fraud.db")
MODEL_DIR=os.getenv("MODEL_DIR","artifacts/models")
DATA_PATH=os.getenv("DATA_PATH","data/raw/transactions.csv")
KAFKA_BOOTSTRAP_SERVERS=os.getenv("KAFKA_BOOTSTRAP_SERVERS","localhost:9092")
KAFKA_TOPIC=os.getenv("KAFKA_TOPIC","transactions")
API_URL=os.getenv("API_URL","http://127.0.0.1:8000")
