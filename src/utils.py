from sklearn.metrics import classification_report,roc_auc_score



def publish_model_socres(X,y,cv_fit,plotting=True):
    y_pred=cv_fit.predict(X)
    y_pred_prob = cv_fit.predict_proba(X)
    print(classification_report(y,y_pred))
    print("ROC_AUC Score",roc_auc_score(y,y_pred_prob[:,1]))