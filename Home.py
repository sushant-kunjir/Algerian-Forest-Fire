import streamlit as st
import pandas as pd 
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Algerian Forest Fire Project",
    layout="wide"
) 
col1, col2 = st.columns([8,2])

with col2:
    if st.button(
        "🔥 Predict FWI",
        use_container_width=True,
        type="primary"
    ):
        st.switch_page("https://sushant-algerian-forest-project3.streamlit.app/prediction")
st.title("Algerian Forest Fire Analysis")
st.markdown("""
    This Project shows the Analysis of Algerian forest 
    """)

st.link_button(
    "🔮 Prediction",
    "https://algerian-forest-fire-2-fgvw.onrender.com/"
)

df=pd.read_csv("Algerian_forest_fire_cleaned_dataset.csv")
np.round(df,2)


df['Classes']=np.where(df['Classes'].str.contains('not fire'),'not fire','fire')

df["Date"] = pd.to_datetime(
    {
        "year": df["year"],
        "month": df["month"],
        "day": df["day"]
    }
)
df['Date']=df['Date'].astype('datetime64[ns]')



st.sidebar.title('Basic Information')
val={
    'RH':'Relative Humidity',
    'WS':'Wind Speed',
    'FFMC':'Fine Fuel Moisture Code',
    'DMC':'Duff Moisture Code',
    'DC':'Drought Code',
    'ISI':'Initial Spread Index',
    'BUI':'Build-up Index',
    'FWI':'Fire Weather Index',
    'Classes':'Fire and Not Fire',
    'Region':'Bejaia Region and Sidi-Bel Abbes Region'
}
st.sidebar.table(val)

st.sidebar.header("Filters")
Region=st.sidebar.multiselect(
    """Select Region -

    0:Bajaia Region,

    1:Sidi-Bel Abbes Region 
    """,
    options=df['Region'].unique(),
    default=df['Region'].unique()
)

Class=st.sidebar.multiselect(
    'Select Class',
    options=df['Classes'].unique(),
    default=df['Classes'].unique()
)
date1 = df["Date"].min()
date2 = df["Date"].max()

date_range = st.sidebar.slider(
    "Select Date Range",
    min_value=date1.to_pydatetime(),
    max_value=date2.to_pydatetime(),
    value=(date1.to_pydatetime(), date2.to_pydatetime())
)
start_date, end_date = date_range

filtered_df = df[
    (df['Region'].isin(Region)) &
    (df['Classes'].isin(Class)) &
    (df["Date"] >= pd.Timestamp(start_date)) &
    (df["Date"] <= pd.Timestamp(end_date))
]
# Separate data by region
bejaia = filtered_df[filtered_df["Region"] == 0]
sidi = filtered_df[filtered_df["Region"] == 1]

# Temperature
temp_bejaia = bejaia["Temperature"].mean()
temp_sidi = sidi["Temperature"].mean()

# Humidity
hum_bejaia = bejaia["RH"].mean()
hum_sidi = sidi["RH"].mean()

# Wind Speed
wind_bejaia = bejaia["Ws"].mean()
wind_sidi = sidi["Ws"].mean()

# Rain
rain_bejaia = bejaia["Rain"].mean()
rain_sidi = sidi["Rain"].mean()

st.subheader("📊 Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Avg. Temp. of Bejaia Region",
    f"{temp_bejaia:.2f} °C"
)

col2.metric(
    "Avg. Temp. of Sidi Bel Region",
    f"{temp_sidi:.2f} °C"
)

col3.metric(
    "Avg. Humidity of Bejaia Region",
    f"{hum_bejaia:.2f}%"
)

col4.metric(
    "Avg. Humidity of Sidi Bel Region",
    f"{hum_sidi:.2f}%"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Avg. Wind Speed of Bejaia Region",
    f"{wind_bejaia:.2f} km/hr"
)

col2.metric(
    "Avg. Wind Speed of Sidi Bel Region",
    f"{wind_sidi:.2f} km/hr"
)

col3.metric(
    "Avg. Rain of Bejaia Region",
    f"{rain_bejaia:.2f} mm"
)

col4.metric(
    "Avg. Rain of Sidi Bel Region",
    f"{rain_sidi:.2f} mm"
)


# Separate data by region
bejaia_df = filtered_df[filtered_df["Region"] == 0]
sidi_df = filtered_df[filtered_df["Region"] == 1]

# Count Fire vs Not Fire
bejaia_pie = (
    bejaia_df.groupby("Classes")
    .size()
    .reset_index(name="Count")
)

sidi_pie = (
    sidi_df.groupby("Classes")
    .size()
    .reset_index(name="Count")
)

# Display side by side
col1, col2 = st.columns(2)

with col1:
    fig1 = px.pie(
        bejaia_pie,
        names="Classes",
        values="Count",
        title="🔥 Fire Distribution - Bejaia Region",
        color_discrete_sequence=px.colors.sequential.Purp_r
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    fig2 = px.pie(
        sidi_pie,
        names="Classes",
        values="Count",
        title="🔥 Fire Distribution - Sidi Bel-Abbes Region",
        color_discrete_sequence=px.colors.sequential.Purp_r


    )
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("🗂 View Dataset")

st.dataframe(filtered_df)

# -----------------------------
# DOWNLOAD OPTION
# -----------------------------
csv = filtered_df.to_csv(index=False).encode('utf-8')

st.download_button(
    "Download Filtered Data",
    data=csv,
    file_name='filtered_Algerian_forest_fire.csv',
    mime='text/csv'
)