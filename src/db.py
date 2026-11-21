from sqlalchemy import create_engine,Column,Integer,String,Float,Text,DateTime
from sqlalchemy.orm import declarative_base,sessionmaker
from datetime import datetime
from src.utils.config import DATABASE_URL
engine=create_engine(DATABASE_URL,connect_args={"check_same_thread":False} if DATABASE_URL.startswith("sqlite") else {})
Session=sessionmaker(bind=engine); Base=declarative_base()
class Tx(Base):
 __tablename__="transactions"
 id=Column(Integer,primary_key=True); transaction_id=Column(String(100),unique=True,index=True)
 amount=Column(Float); step=Column(Integer); risk_score=Column(Float); risk_category=Column(String(30))
 explanation=Column(Text); created_at=Column(DateTime,default=datetime.utcnow)
def init_db(): Base.metadata.create_all(engine)
