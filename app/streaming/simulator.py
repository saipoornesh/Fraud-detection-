import argparse,random,time,uuid,json
from src.utils.config import API_URL,KAFKA_BOOTSTRAP_SERVERS,KAFKA_TOPIC
def tx(): return {"transaction_id":str(uuid.uuid4()),"amount":round(random.lognormvariate(4,1),2),
"step":int(time.time()),"transaction_type":random.choice(["PAYMENT","TRANSFER","CASH_OUT"])}
def main():
 p=argparse.ArgumentParser(); p.add_argument("--api",action="store_true"); p.add_argument("--count",type=int,default=20); a=p.parse_args()
 if a.api:
  import requests
  for _ in range(a.count): requests.post(API_URL+"/predict",json=tx(),timeout=20); time.sleep(.5)
 else:
  from kafka import KafkaProducer
  k=KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,value_serializer=lambda x:json.dumps(x).encode())
  for _ in range(a.count): k.send(KAFKA_TOPIC,tx()); time.sleep(.5)
  k.flush()
if __name__=="__main__": main()
