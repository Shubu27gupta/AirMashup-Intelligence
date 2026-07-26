import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌍",
    layout="wide"
)

# -------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------

st.markdown("""
<style>

#MainMenu{visibility:hidden;}
footer{visibility:hidden;}

.stApp{
    background-color:#f8f9fa;
}

h1,h2,h3{
    color:#0f172a;
}

div[data-testid="metric-container"]{
    background:white;
    border-radius:12px;
    padding:15px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.1);
}

.stButton>button{
    background:#2563eb;
    color:white;
    width:100%;
    height:50px;
    border:none;
    border-radius:10px;
    font-size:18px;
}

</style>
""",unsafe_allow_html=True)

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------

@st.cache_resource
def load_files():

    model=joblib.load("aqi_model.pkl")

    scaler=joblib.load("scaler.pkl")

    features=joblib.load("features.pkl")

    return model,scaler,features

model,scaler,features=load_files()

# -------------------------------------------------------
# LOAD DATASET
# -------------------------------------------------------

uploaded_file=st.sidebar.file_uploader(
    "Upload CSV or Excel",
    type=["csv","xlsx"]
)

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):

        df=pd.read_csv(uploaded_file)

    else:

        df=pd.read_excel(uploaded_file)

else:

    if os.path.exists("city_day.csv"):

        df=pd.read_csv("city_day.csv")

    else:

        df=pd.DataFrame()

# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------

st.sidebar.title("🌍 AQI Dashboard")

page=st.sidebar.radio(

"Menu",

[
"🏠 Home",
"📊 Dataset Explorer",
"📈 Visualization",
"🤖 AQI Prediction"
]

)

# -------------------------------------------------------
# HOME PAGE
# -------------------------------------------------------

if page=="🏠 Home":

    st.title("🌍 Air Quality Index Prediction System")

    st.write(
    """
This project predicts the Air Quality Index (AQI)
using a Random Forest Regression model.
    """
    )

    st.divider()

    c1,c2,c3,c4=st.columns(4)

    c1.metric("Rows",df.shape[0])

    c2.metric("Columns",df.shape[1])

    if len(df)>0:

        c3.metric("Missing",int(df.isnull().sum().sum()))

        c4.metric("Duplicates",int(df.duplicated().sum()))

    st.divider()

    col1,col2=st.columns([2,1])

    with col1:

        st.subheader("Project Objective")

        st.write("""

• Predict AQI

• Analyze pollution data

• Visualize pollutants

• Health Recommendation

• Machine Learning Model

        """)

    with col2:

        fig=go.Figure(go.Indicator(

            mode="gauge+number",

            value=150,

            title={'text':"Sample AQI"},

            gauge={

                'axis':{'range':[0,500]},

                'bar':{'color':'green'}

            }

        ))

        st.plotly_chart(fig,use_container_width=True)

    st.success("Use the left sidebar to explore the dashboard.")
# ==========================================================
# DATASET EXPLORER
# ==========================================================

elif page=="📊 Dataset Explorer":

    st.title("📊 Dataset Explorer")

    if df.empty:

        st.error("Please upload a CSV or Excel file.")

    else:

        col1,col2,col3,col4=st.columns(4)

        col1.metric("Rows",df.shape[0])
        col2.metric("Columns",df.shape[1])
        col3.metric("Missing",int(df.isnull().sum().sum()))
        col4.metric("Duplicates",int(df.duplicated().sum()))

        st.divider()

        st.subheader("Dataset Preview")

        rows=st.slider(
            "Number of Rows",
            5,
            min(100,len(df)),
            10
        )

        st.dataframe(
            df.head(rows),
            use_container_width=True
        )

        st.divider()

        st.subheader("Column Information")

        info=pd.DataFrame({

            "Column":df.columns,

            "Datatype":df.dtypes.astype(str),

            "Missing":df.isnull().sum().values,

            "Unique":df.nunique().values

        })

        st.dataframe(
            info,
            use_container_width=True
        )

        st.divider()

        st.subheader("Statistical Summary")

        numeric=df.select_dtypes(include=np.number)

        if len(numeric.columns)>0:

            st.dataframe(
                numeric.describe().T,
                use_container_width=True
            )

        st.divider()

        st.subheader("Missing Values")

        miss=pd.DataFrame({

            "Column":df.columns,

            "Missing Values":df.isnull().sum()

        })

        st.dataframe(
            miss,
            use_container_width=True
        )

        fig=px.bar(

            miss,

            x="Column",

            y="Missing Values",

            color="Missing Values",

            title="Missing Values"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        st.subheader("Download Dataset")

        csv=df.to_csv(index=False).encode()

        st.download_button(

            "📥 Download CSV",

            csv,

            "dataset.csv",

            "text/csv"

        )
# ==========================================================
# VISUALIZATION
# ==========================================================

elif page == "📈 Visualization":

    st.title("📈 Data Visualization")

    if df.empty:

        st.error("Please upload a dataset first.")

    else:

        numeric = df.select_dtypes(include=np.number).columns.tolist()

        category = df.select_dtypes(exclude=np.number).columns.tolist()

        if len(numeric) > 0:

            st.subheader("Histogram")

            column = st.selectbox(
                "Select Numerical Column",
                numeric
            )

            fig = px.histogram(
                df,
                x=column,
                nbins=40,
                color_discrete_sequence=["royalblue"]
            )

            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            st.subheader("Box Plot")

            fig = px.box(
                df,
                y=column,
                color_discrete_sequence=["orange"]
            )

            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            st.subheader("Scatter Plot")

            x = st.selectbox(
                "X Axis",
                numeric,
                key="scatter_x"
            )

            y = st.selectbox(
                "Y Axis",
                numeric,
                index=min(1, len(numeric)-1),
                key="scatter_y"
            )

            fig = px.scatter(
                df,
                x=x,
                y=y,
                color=y
            )

            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            st.subheader("Correlation Heatmap")

            corr = df[numeric].corr()

            fig = px.imshow(
                corr,
                text_auto=True,
                color_continuous_scale="Viridis"
            )

            st.plotly_chart(fig, use_container_width=True)

        if len(category) > 0:

            st.divider()

            st.subheader("Bar Chart")

            cat = st.selectbox(
                "Select Category",
                category
            )

            value = df[cat].value_counts().reset_index()

            value.columns = [cat, "Count"]

            fig = px.bar(
                value,
                x=cat,
                y="Count",
                color="Count"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            st.subheader("Pie Chart")

            fig = px.pie(
                value,
                names=cat,
                values="Count",
                hole=0.4
            )

            st.plotly_chart(fig, use_container_width=True)
# ==========================================================
# AQI PREDICTION
# ==========================================================

elif page == "🤖 AQI Prediction":

    st.title("🤖 AQI Prediction")

    st.write("Enter the pollutant values to predict AQI.")

    col1, col2 = st.columns(2)

    with col1:
        pm25 = st.number_input("PM2.5", min_value=0.0, value=50.0)
        pm10 = st.number_input("PM10", min_value=0.0, value=80.0)
        no = st.number_input("NO", min_value=0.0, value=20.0)
        no2 = st.number_input("NO2", min_value=0.0, value=30.0)
        nox = st.number_input("NOx", min_value=0.0, value=40.0)
        nh3 = st.number_input("NH3", min_value=0.0, value=20.0)

    with col2:
        co = st.number_input("CO", min_value=0.0, value=1.0)
        so2 = st.number_input("SO2", min_value=0.0, value=10.0)
        o3 = st.number_input("O3", min_value=0.0, value=40.0)
        benzene = st.number_input("Benzene", min_value=0.0, value=5.0)
        toluene = st.number_input("Toluene", min_value=0.0, value=10.0)

        season = st.selectbox(
            "Season",
            [
                "Monsoon",
                "Post-Monsoon",
                "Summer",
                "Winter"
            ]
        )

    # Season Encoding
    season_dict = {
        "Monsoon": 0,
        "Post-Monsoon": 1,
        "Summer": 2,
        "Winter": 3
    }

    season_enc = season_dict[season]

    if st.button("Predict AQI"):

        input_data = np.array([[
            pm25,
            pm10,
            no,
            no2,
            nox,
            nh3,
            co,
            so2,
            o3,
            benzene,
            toluene,
            season_enc
        ]])

        try:

            input_scaled = scaler.transform(input_data)

            prediction = model.predict(input_scaled)[0]

            st.success(f"Predicted AQI : {prediction:.2f}")

            if prediction <= 50:
                category = "🟢 Good"
                advice = "Air quality is good."

            elif prediction <= 100:
                category = "🟡 Satisfactory"
                advice = "Air quality is acceptable."

            elif prediction <= 200:
                category = "🟠 Moderate"
                advice = "Sensitive people should reduce outdoor activity."

            elif prediction <= 300:
                category = "🔴 Poor"
                advice = "Wear a mask while going outside."

            elif prediction <= 400:
                category = "🟣 Very Poor"
                advice = "Avoid outdoor activities."

            else:
                category = "⚫ Severe"
                advice = "Stay indoors and wear an N95 mask if going outside."

            st.subheader("AQI Category")
            st.info(category)

            st.subheader("Health Recommendation")
            st.success(advice)

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=float(prediction),
                title={"text": "Predicted AQI"},
                gauge={
                    "axis": {"range": [0, 500]},
                    "bar": {"color": "green"}
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction Error: {e}")
