import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score, make_scorer
from sklearn.model_selection import KFold
import streamlit as st
import duckdb
import numpy as np
import plotly.express as px
from pymongo import MongoClient
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from google import genai

gemini_client = genai.Client(api_key=st.secrets["YOUR_API_KEY"])


@st.cache_data
def load_data():
    df = pd.read_excel("Raw_Data.xlsx")
    nosql = df.to_dict(orient="records")
    return df, nosql
df, nosql = load_data()


# Connect MongoDB
client = MongoClient(st.secrets["MONGO_URI"])
db = client["mydatabase"]
collection = db["mycollection"]
if "nosql_update" not in st.session_state:
    collection.delete_many({})
    if nosql:
        collection.insert_many(nosql)
    st.session_state["nosql_update"] = True

st.title("DADS5001_Project : EV car Parking analysis :car:")
st.markdown("**📌📌 จุดประสงค์ : วิเคราะห์พฤติกรรมการใช้สถานีอัดประจุยานยนต์ไฟฟ้า โดยจะวิเคราะห์หากลุ่มผู้ใช้งานที่มีแนวโน้มในการจอดแช่ / จอดเนียน " \
"โดยใช้ ML เข้ามาช่วยในการจัดกลุ่มผู้ใช้งาน 🏆🏅**")
st.header("EV car parking report :")
st.markdown("*📓 ข้อมูลการใช้งานสถานีอัดประจุยานยนต์ไฟฟ้าตามสถานที่ต่าง ๆ*")
st.dataframe(df, hide_index=True)

# Transform dataframe
st.header("Features Selection + Add new Features : ข้อมูลเฉพาะที่เกี่ยวข้องกับการวิเคราะห์")
st.markdown("*✖️ Features ที่ทำการตัด จะเป็นส่วนที่ไม่เกี่ยวข้อง เช่น ชื่อ-นามสกุล, E-mail, ค่าใช้จ่ายในการใช้บริการ เป็นต้น*")
st.markdown("*✔️ Features ที่ทำการเพิ่มคือ พลังงานไฟฟ้าที่ได้รับต่อ 1 ชั่วโมงของ transection นั้นๆ (Energyrate) ," \
"พลังงานไฟฟ้าต่ำสุดและสูงสุดที่เครื่องชาร์จนั้นๆ ต่อ 1 ชั่วโมง (min-max kWh), ค่าพลังงานเปรียบเทียบระหว่าง Energyrate กับ min_kWh " \
"เพื่อให้ Model ได้เรียนรู้*")

df["UsingTime"] = pd.to_timedelta(df["Using Time (HH:MM:SS)"]).dt.total_seconds()
df["Time (Unit Hour)"] = df["UsingTime"] / 3600
df["Energyrate"] = df["Charging Amount (kWh)"] / df["Time (Unit Hour)"]
min_condition = [
    df["Charging Speed (kW)"] <= 22,
    df["Charging Speed (kW)"] <=50,
    df["Charging Speed (kW)"] > 50,
]
min_chioce = [
    3.5,#*df["Time (Unit Hour)"],
    df["Charging Speed (kW)"],#*df["Time (Unit Hour)"],
    df["Charging Speed (kW)"]#*df["Time (Unit Hour)"] /2
]
df["Minimium_output"] = np.select(min_condition,min_chioce)


max_condition = [
    df["Charging Speed (kW)"] == 7,
    df["Charging Speed (kW)"] == 22,
    df["Charging Speed (kW)"] <=50,
    df["Charging Speed (kW)"] > 50,
]
max_chioce = [
    df["Charging Speed (kW)"],#*df["Time (Unit Hour)"],
    11,#*df["Time (Unit Hour)"],
    df["Charging Speed (kW)"],#*df["Time (Unit Hour)"],
    df["Charging Speed (kW)"]#*df["Time (Unit Hour)"] /2
]
df["Maximium_output"] = np.select(max_condition,max_chioce)
df["diff"] = df["Energyrate"] - df["Minimium_output"]
AC_data = df[(df["Charger Type"]=="AC")]

# Prepairation Training Dataset

data = AC_data[[
    "Time (Unit Hour)",
    "Charger Type",
    "Charging Speed (kW)",
    "Charging Amount (kWh)",
    "Energyrate",
    "Minimium_output",
    "Maximium_output",
    "diff",
]]
st.dataframe(data)

# Scatter plot to Check outlier
st.header("Scatter plot : กราฟแสดงความสัมพันธ์ระหว่างระยะเวลากับพลังงานไฟฟ้าที่ได้รับ")
st.markdown("*💬 จะสังเกตุได้ว่าข้อมูลมี outlier อยู่ และประกอบกับพฤติกรรมการชาร์จรถไฟฟ้าในสถานีฯ ไม่ควรใช้ระยะเวลาในการใช้งานเกิน 10 ชั่วโมง (อ้างอิงตามระยะเวลาเปิด-ปิดสถานีฯ)*")
if "fig1_state" not in st.session_state:
    fig1 = px.scatter(
        data_frame= data,
        x="Time (Unit Hour)",
        y ='Charging Amount (kWh)',
        title="Duration vs kWh"
        )
    st.session_state["fig1_state"] = fig1

st.plotly_chart(st.session_state["fig1_state"])



# Remove outlier
AC_filter = data[(data["Time (Unit Hour)"]<=10)]

# Scatter plot after remove outlier
st.header("Scatter plot : กราฟแสดงความสัมพันธ์ระหว่างระยะเวลากับพลังงานไฟฟ้าที่ได้รับ หลังนำข้อมูล outlier ออก (>10 hr)")
st.markdown("*💬 จะได้ข้อมูลเพื่อนำไปเข้าสู่กระบวณการ Clustering เพื่อนำมาวิเคราะห์การใช้งาน*")
if "fig2_state" not in st.session_state:
    fig2 = px.scatter(
        data_frame= AC_filter,
        x="Time (Unit Hour)",
        y ='Charging Amount (kWh)',
        title="Duration vs kWh"
        )
    st.session_state["fig2_state"] = fig2
st.plotly_chart(st.session_state["fig2_state"])

# ML model (Onehot > PCA > K-Mean)
@st.cache_resource
def ML(AC_filter = AC_filter, AC_data = AC_data):
    numeric_features = selector(dtype_include=['int64', 'float64'])(AC_filter)
    categorical_features = selector(dtype_include=['object'])(AC_filter)


    preprocessor = ColumnTransformer(
        transformers=[        
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )

    pipe = Pipeline([
        ('preprocess', preprocessor),
        ('pca', PCA()),
        ('kmeans', KMeans(random_state=42))
    ])

    def silhouette_scorer(estimator, X):
        X_transformed = estimator.named_steps['preprocess'].transform(X)
        X_transformed = estimator.named_steps['pca'].transform(X_transformed)
        labels = estimator.named_steps['kmeans'].fit_predict(X_transformed)
        return silhouette_score(X_transformed, labels)
    param_grid = {
        'pca__n_components': [1, 2, 3],
        'kmeans__n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        pipe,
        param_grid,
        scoring=silhouette_scorer,
        cv=cv,
        verbose=1
    )
    grid.fit(AC_filter)
    # Apply model with best parameter to alldata
    X_out_transformed = grid.best_estimator_.named_steps['preprocess'].transform(AC_data)
    X_out_pca = grid.best_estimator_.named_steps['pca'].transform(X_out_transformed)
    AC_data['cluster'] = grid.best_estimator_.named_steps['kmeans'].predict(X_out_pca).astype(str)
    return (AC_data)

AC_data = ML()
st.header("Dataframe : หลัง Clustering 💻")
st.markdown("*💬 ข้อมูลหลังจากใช้ PCA ควบคู่กับ K-mean เพื่อจัดกลุ่มหาพฤติกรรมการใช้งาน โดย Model ได้เลือก Best parameter ผ่าน GridSearchCV ได้ว่า :*")
st.markdown("*PCA = 1 แกน และ K-mean ได้ 2 กลุ่ม*")
st.dataframe(AC_data, hide_index=True)


nosql_AC_data= AC_data.to_dict(orient="records")
collect_result = db["result"]
if "nosql_AC_data" not in st.session_state:
    collect_result.delete_many({})
    if nosql_AC_data:
        collect_result.insert_many(nosql_AC_data)
    st.session_state["nosql_AC_data"] = True


# Scatter plot to see detail
st.header("Scatter plot : กราฟแสดงความสัมพันธ์ระหว่างระยะเวลากับพลังงานไฟฟ้าที่ได้รับ หลังแบ่งกลุ่ม")
st.markdown("*💬 เมื่อพิจารณาแล้วพบว่า กลุ่มที่ 1 จะเป็นกลุ่มที่มีระยะเวลาในการชาร์จนานแต่กลับได้พลังงานไฟฟ้าต่ำเมื่อเปรียบเทียบกับกลุ่มที่ 0*")
st.markdown("*💬 ดังนั้นอาจจะสรุปได้ว่า กลุ่มที่ 1 มีแนวโน้มที่จะเป็นลูกค้าที่นำรถเข้ามาใช้บริการแต่ไม่ได้ชาร์จไฟ (จอดเนียน)*")
if "fig3_state" not in st.session_state:
    fig3 = px.scatter(
        data_frame= AC_data,
        x="Time (Unit Hour)",
        y ='Charging Amount (kWh)',
        color="cluster",
        color_discrete_sequence=px.colors.qualitative.Plotly,
        hover_data = ["cluster"],
        title="Duration vs kWh"
        )

    fig3.update_layout(
        xaxis=dict(range=[0, 10]),
    )
    st.session_state["fig3_state"] = fig3
st.plotly_chart(st.session_state["fig3_state"])

#Analyst data with SQL 
st.header("Data analysis with SQL (duckdb) : วิเคราะห์การใช้งานตามกลุ่ม Cluster") 
result = duckdb.query("""
                      SELECT "cluster", "Charging Speed (kW)", avg("Time (Unit Hour)"), avg("Charging Amount (kWh)")
                      FROM AC_data 
                      GROUP BY "cluster", "Charging Speed (kW)" """).to_df()
result = result.sort_values(by="cluster").reset_index(drop=True)

def SQL_to_Text(result):
    text_sql = []
    for _,row in result.iterrows():
        text = (
            f"กลุ่ม:{int(row['cluster'])} ,"
            f"ขนาดเครื่องชาร์จ:{row['Charging Speed (kW)']}kW ,"
            f"เวลาในการชาร์จเฉลี่ย:{float(row['avg(\"Time (Unit Hour)\")']):.2f}hour ,"
            f"พลังงงานไฟฟ้าที่ได้รับ:{float(row['avg(\"Charging Amount (kWh)\")']):.2f}kWh"
        )
        text_sql.append(text)
    return "\n".join(text_sql)
text_sql = SQL_to_Text(result)

prompt_sql = f"""สวัสดี ฉันอยากให้คุณช่วยวิเคราะห์เรื่องการใช้งานสถานีชาร์จรถยนต์ไฟฟ้าหน่อย 
โดยที่ฉันต้องการจะหาว่ากลุ่มไหนคือกลุ่มที่ มีแนวโน้มจะแอบเข้ามาจอดโดยที่ไม่ได้ใช้บริการ (จอดแช่/จอดเนียน)
โดยฉันมีข้อมูลแต่กลุ่มดังนี้ {text_sql}"""



@st.cache_data
def callAPI_SQL(prompt_sql):
    response_sql = gemini_client.models.generate_content(
    model="gemini-2.0-flash", contents=prompt_sql)
    return response_sql

if "AI_SQL" not in st.session_state:
    st.session_state["AI_SQL"] = None

use_ai_SQL = st.checkbox("🔍 วิเคราะห์โดย Gemini AI",key = "use_ai_SQL")
if use_ai_SQL:
    if st.session_state["AI_SQL"] == None:
        st.session_state["AI_SQL"] = callAPI_SQL(prompt_sql)
    st.write(st.session_state["AI_SQL"].text)
else:
    st.markdown("*💬จะเห็นได้ชัดเจนขึ้นเลยว่า กลุ่มที่ 1 เมื่อใช้เวลาที่ใกล้เคียงกัน แต่พลังงานที่ได้รับกลับน้อยกว่ามากพอสมควร*")


if "fig4_state" not in st.session_state:  
    fig4 = px.line(result,
                x='Charging Speed (kW)',
                y='avg("Charging Amount (kWh)")',
                hover_data=["cluster"],
                color="cluster",
                color_discrete_sequence=px.colors.qualitative.Plotly)
    st.session_state["fig4_state"] = fig4

st.plotly_chart(st.session_state["fig4_state"])
st.dataframe(result, hide_index=True)


#Analyst data with SQL 

pipe = [
    {
     "$group" : {
         "_id": {"Cluster" : "$cluster",
                 "Group" : "$Group"
                 },
         "count" : {"$sum":1}
        }
    },
    {
        "$project" : {
            "Cluster" : "$_id.Cluster",
            "Group" : "$_id.Group",
            "count" : 1,
            "_id":0
        }
    }
]
nosql_result = pd.DataFrame(list(collect_result.aggregate(pipe)))
nosql_result = nosql_result[["Cluster","Group","count"]]
per_cluster = nosql_result.groupby("Group")["count"].transform("sum")
nosql_result.sort_values(by="Group", inplace=True)

st.header("Data analysis with NoSQL (mongodb) : วิเคราะห์การใช้งานตามกลุ่ม Cluster")

groups = nosql_result['Group'].unique()
if "fig5_state" not in st.session_state:
    fig5 = make_subplots(rows=1, 
                        cols=len(groups), 
                        specs=[[{'type': 'domain'}]*len(groups)],
                        subplot_titles=[f"Group: {g}" for g in groups],
                        color_discrete_sequence=px.colors.qualitative.Plotly,
                        sort = FALSE
                        )
####
    for i, group in enumerate(groups):
        df_group = nosql_result[nosql_result['Group'] == group]

        fig5.add_trace(go.Pie(
            labels=df_group['Cluster'],
            values=df_group['count'],
            name=f"Group {group}",
            hoverinfo="label+percent+value"
        ), row=1, col=i+1)
    st.session_state["fig5_state"] = fig5

@st.cache_data
def NoSql_to_Text(nosql_result):
    text_NoSql = []
    for _,row in nosql_result.iterrows():
        text = (
            f"กลุ่ม:{int(row['Cluster'])} ,"
            f"กลุ่มสถานที่:{row['Group']} ,"
            f"จำนวน:{int(row["count"])}"
        )
        text_NoSql.append(text)
    return "\n".join(text_NoSql)
        
text_NoSql = NoSql_to_Text(nosql_result)
prompt_NoSql = f"""สวัสดี ฉันอยากให้คุณช่วยวิเคราะห์เรื่องการใช้งานสถานีชาร์จรถยนต์ไฟฟ้าหน่อย 
โดยฉันมีข้อมูลที่ถูกแบ่งแยกไว้แล้ว 2 กลุ่ม คือ กลุ่ม 0 จะเป็นกลุ่มจอดชาร์จปกติ และ กลุ่ม 1 จะเป็นกลุ่มที่จอดแช่/จอดเนียน 
ฉันอยากให้คุณวิเคราะห์พฤติกรรมการใช้งานโดยคำนึงปัจจัยกลุ่มสถานที่เป็นหลัก โดยฉันมีข้อมูลดังนี้ {text_NoSql}"""
        
@st.cache_data
def callAPI_NoSQL(prompt_NoSql):
    response_NoSql = gemini_client.models.generate_content(
    model="gemini-2.0-flash", contents=prompt_sql)
    return response_NoSql

if "AI_NoSql" not in st.session_state:
    st.session_state["AI_NoSql"] = None

use_ai_Nosql = st.checkbox("🔍 วิเคราะห์โดย Gemini AI",key="use_ai_Nosql")
if use_ai_Nosql:
    if st.session_state["AI_NoSql"] == None:
        st.session_state["AI_NoSql"] = callAPI_SQL(prompt_NoSql)
    st.write(st.session_state["AI_NoSql"].text)
else:
    st.markdown("*💬 พิจารณาจากกลุ่มผู้ใช้งานโดยแบ่งตาม Group จะพบว่า Group Office จะมีอัตราการจอดแช่มากที่สุด" \
    "ซึ่งสอดคล้องกับพฤติกรรมของกลุ่มนี้ เนื่องจากจะเป็นกลุ่มผู้ใช้งานรถยนต์ส่วนกลางของบริษัทเป็นหลัก และมักจะจอดแช่ไว้ในสถานีฯ หลังใช้งานเสร็จ*")

st.plotly_chart(st.session_state["fig5_state"])
st.dataframe(nosql_result, hide_index=True)


