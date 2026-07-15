from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application=Flask(__name__)
app=application

ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standard_scalar=pickle.load(open('models/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        Temprature=float(request.form.get("Temprature"))
        RH=float(request.form.get("RH"))
        WS=float(request.form.get("WS"))
        Rain=float(request.form.get("Rain"))
        FFMC=float(request.form.get("FFMC"))
        DMC=float(request.form.get("DMC"))
        ISI=float(request.form.get("ISI"))
        Classes=float(request.form.get("Classes"))
        Region=float(request.form.get("Region"))

        new_scaled_data=standard_scalar.transform([[Temprature,RH,WS,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_scaled_data)

        return render_template('algerian-forest-fire-v152.vercel.app',results=np.round(result[0],2))
    else:
        return render_template('algerian-forest-fire-v152.vercel.app')

if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0",port=8000)