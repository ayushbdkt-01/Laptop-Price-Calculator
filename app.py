import streamlit as st
import pickle as pkl
import numpy as np

st.title("Laptop Price Calculator 🚀")
st.header("Made with ❤️ by Ayush")

pipe = pkl.load(open(pipe.pkl, 'rb'))
df = pkl.load(open(df.pkl, 'rb'))

company = st.selectbox("Brand", df['Company'].unique())
type = st.selectbox("Type of Laptop", df['TypeName'].unique())
ram = st.selectbox("Ram Size [in GB]", [2,4,6,8,16,24,32,64])
weight = st.number_input("Weight of Laptop")
touchscreen = st.selectbox("TouchScreen", ['Yes', 'No'])
ips = st.selectbox("Ips", ['Yes', 'No'])
screen_size = st.number_input('screenSize')
resolution = st.selectbox('Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
cpu = st.selectbox("CPU", df['CpuBrand'].unique())
hdd = st.selectbox("HDD in GB", [0,64,128,256,512,1024,2048])
ssd = st.selectbox("SSD in GB", [0,64,128,256,512,1024,2048])
gpu = st.selectbox("GPU", df['GpuBrand'].unique())
os = st.selectbox("OS", df['os'].unique())

if st.button("Predict Price"):

    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])

    ppi= ((x_res**2)+(y_res)**2)**0.5/screen_size

    if touchscreen=="Yes": touchscreen=1
    else: touchscreen=0

    if ips=="Yes": ips=1
    else: ips=0

    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1,12)

    st.title('Predicted Price: ₹')
    st.title(int(np.exp(pipe.predict(query))[0]))
