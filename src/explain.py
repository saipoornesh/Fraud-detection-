def shap_explanation(model,background,row,feature_names):
 try:
  import shap
  e=shap.Explainer(model,background); v=e(row).values[0]
  ids=sorted(range(len(v)),key=lambda i:abs(float(v[i])),reverse=True)[:5]
  return [{"feature":feature_names[i],"contribution":float(v[i])} for i in ids]
 except Exception as exc: return {"message":f"SHAP unavailable: {exc}"}
def lime_explanation(model,training,row,feature_names):
 try:
  from lime.lime_tabular import LimeTabularExplainer
  e=LimeTabularExplainer(training,feature_names=feature_names,class_names=["genuine","fraud"],mode="classification")
  return e.explain_instance(row,model.predict_proba,num_features=5).as_list()
 except Exception as exc: return {"message":f"LIME unavailable: {exc}"}
