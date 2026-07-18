
# Prediction.py
import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
from pathlib import Path


st.set_page_config(page_title="FWI Prediction", page_icon="🔥", layout="wide")

col1, col2 = st.columns([8,2])

with col2:
    if st.button("⬅ Dashboard"):
        st.page_link("https://sushant-algerian-forest-project3.streamlit.app/")

st.markdown("""
<style>
.stApp{background:#15110d;color:white;}
.block-container{padding-top:1.2rem;}
.title{font-size:42px;font-weight:800;color:#ff6b2d;}
.sub{color:#c7c7c7;}
.card{background:#211912;padding:22px;border-radius:16px;border:1px solid #3d2c1f;}
.stButton>button{width:100%;height:52px;background:#ff6b2d;color:white;font-weight:bold;border:none;border-radius:10px;}
</style>
""", unsafe_allow_html=True)

base = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()
model_path = base/"models"/"ridge.pkl"
scaler_path = base/"models"/"scaler.pkl"

with open(model_path,"rb") as f:
    model = pickle.load(f)
with open(scaler_path,"rb") as f:
    scaler = pickle.load(f)

st.markdown("<div class='title'>🔥 Forest Fire Weather Index Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='sub'>AI Powered Ridge Regression Model</div>", unsafe_allow_html=True)
st.divider()

left,right = st.columns([3,2])

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🌤 Atmospheric Conditions")
    c1,c2,c3=st.columns(3)
    with c1:
        t=st.number_input("Temperature",0.0,60.0,29.0)
    with c2:
        rh=st.number_input("Humidity",0.0,100.0,45.0)
    with c3:
        ws=st.number_input("Wind Speed",0.0,100.0,15.0)
    rain=st.number_input("Rain",0.0,500.0,0.0)

    st.subheader("🔥 Fuel Behaviour Codes")
    c1,c2,c3=st.columns(3)
    with c1:
        ffmc=st.number_input("FFMC",0.0,100.0,85.0)
    with c2:
        dmc=st.number_input("DMC",0.0,300.0,26.0)
    with c3:
        isi=st.number_input("ISI",0.0,50.0,5.0)

    region = 0 if st.selectbox("Region",["Bejaia","Sidi-Bel Abbes"])=="Bejaia" else 1
    classes = 1 if st.selectbox("Classes",["Fire","Not Fire"])=="Fire" else 0

    run = st.button("🔥 Run Prediction")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📊 Prediction")

    if run:
        X=np.array([[t,rh,ws,rain,ffmc,dmc,isi,region,classes]])
        pred=float(model.predict(scaler.transform(X))[0])

        fig=go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            title={"text":"FWI Score"},
            gauge={
                "axis":{"range":[0,30]},
                "steps":[
                    {"range":[0,5],"color":"green"},
                    {"range":[5,12],"color":"yellow"},
                    {"range":[12,21],"color":"orange"},
                    {"range":[21,30],"color":"red"}
                ],
                "bar":{"color":"darkred"}
            }
        ))
        st.plotly_chart(fig,use_container_width=True)
        st.metric("Predicted FWI",f"{pred:.2f}")
        st.progress(min(pred/30,1.0))
        if pred<5:
            st.success("🟢 Low Fire Risk")
        elif pred<12:
            st.warning("🟡 Moderate Fire Risk")
        elif pred<21:
            st.error("🟠 High Fire Risk")
        else:
            st.error("🔴 Extreme Fire Risk")
        st.subheader("Input Summary")
        st.dataframe({
            "Temperature":[t],"RH":[rh],"Ws":[ws],"Rain":[rain],
            "FFMC":[ffmc],"DMC":[dmc],"ISI":[isi],
            "Region":["Bejaia" if region==0 else "Sidi-Bel-Abbes"],
            "Classes":["Fire" if classes else "Not Fire"]
        },hide_index=True)
    else:
        st.info("Enter the inputs and click Run Prediction.")
    st.markdown("</div>", unsafe_allow_html=True)
