import streamlit as st,pandas as pd,plotly.express as px,requests
st.set_page_config(page_title="FinTech Fraud Monitor",layout="wide")
st.title("FinTech Fraud & Anomaly Monitor")
api=st.sidebar.text_input("API URL","http://127.0.0.1:8000")
page=st.sidebar.radio("Page",["Overview","Predict","Transactions & Alerts","Model Metrics"])
def get(path):
 try:
  r=requests.get(api+path,timeout=5); r.raise_for_status(); return r.json()
 except Exception as e: st.error(str(e)); return None
if page=="Overview":
 rows=get("/transactions")
 if rows:
  d=pd.DataFrame(rows); a,b,c=st.columns(3); a.metric("Transactions",len(d))
  b.metric("High risk",int((d.risk_category=="HIGH RISK").sum()))
  c.metric("Medium risk",int((d.risk_category=="MEDIUM RISK").sum()))
  st.plotly_chart(px.histogram(d,x="risk_category"),use_container_width=True); st.dataframe(d)
 else: st.info("No stored transactions yet.")
elif page=="Predict":
 with st.form("tx"):
  amount=st.number_input("Amount",min_value=.01,value=250.); step=st.number_input("Step",min_value=0,value=1)
  typ=st.selectbox("Type",["TRANSFER","PAYMENT","CASH_OUT","CASH_IN","DEBIT"])
  old=st.number_input("Previous balance",min_value=0.,value=1000.); new=st.number_input("New balance",min_value=0.,value=750.)
  go=st.form_submit_button("Analyze")
 if go:
  try:
   r=requests.post(api+"/predict",json={"amount":amount,"step":step,"transaction_type":typ,"oldbalanceOrg":old,"newbalanceOrig":new},timeout=30)
   r.raise_for_status(); st.json(r.json())
  except Exception as e: st.error(str(e))
elif page=="Transactions & Alerts":
 rows=get("/transactions")
 if rows is not None: st.dataframe(pd.DataFrame(rows),use_container_width=True)
 al=get("/alerts")
 if al is not None: st.subheader("Alerts"); st.dataframe(pd.DataFrame(al),use_container_width=True)
else:
 m=get("/metrics")
 if m: st.json(m)
