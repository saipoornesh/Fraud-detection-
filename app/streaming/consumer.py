import json,requests
from kafka import KafkaConsumer
from src.utils.config import KAFKA_BOOTSTRAP_SERVERS,KAFKA_TOPIC,API_URL
def main():
 c=KafkaConsumer(KAFKA_TOPIC,bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,auto_offset_reset="latest",
 value_deserializer=lambda m:json.loads(m.decode()))
 for msg in c:
  t=msg.value
  try: print(requests.post(API_URL+"/predict",json=t,timeout=20).text)
  except requests.RequestException as e: print("API error",e)
if __name__=="__main__": main()
