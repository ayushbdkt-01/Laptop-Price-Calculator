# import streamlit as st
# import pickle as pkl
# import numpy as np

# st.title("Laptop Price Calculator 🚀")
# st.header("Made with ❤️ by Ayush")


# pipe = pkl.load(open('C:\\Users\\ayush\\Desktop\\Laptop-Price-Calculator\\pipe.pkl', 'rb'))
# df = pkl.load(open('C:\\Users\\ayush\\Desktop\\Laptop-Price-Calculator\\df.pkl', 'rb'))

# company = st.selectbox("Brand", df['Company'].unique())
# type = st.selectbox("Type of Laptop", df['TypeName'].unique())
# ram = st.selectbox("Ram Size [in GB]", [2,4,6,8,16,24,32,64])
# weight = st.number_input("Weight of Laptop")
# touchscreen = st.selectbox("TouchScreen", ['Yes', 'No'])
# ips = st.selectbox("Ips", ['Yes', 'No'])
# screen_size = st.number_input('screenSize')
# resolution = st.selectbox('Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])
# cpu = st.selectbox("CPU", df['CpuBrand'].unique())
# hdd = st.selectbox("HDD in GB", [0,64,128,256,512,1024,2048])
# ssd = st.selectbox("SSD in GB", [0,64,128,256,512,1024,2048])
# gpu = st.selectbox("GPU", df['GpuBrand'].unique())
# os = st.selectbox("OS", df['os'].unique())

# if st.button("Predict Price"):

#     x_res = int(resolution.split('x')[0])
#     y_res = int(resolution.split('x')[1])

#     ppi= ((x_res**2)+(y_res)**2)**0.5/screen_size

#     if touchscreen=="Yes": touchscreen=1
#     else: touchscreen=0

#     if ips=="Yes": ips=1
#     else: ips=0

#     query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

#     query = query.reshape(1,12)

#     st.title('Predicted Price: ₹')
#     st.title(int(np.exp(pipe.predict(query))[0]))











import streamlit as st
import pickle as pkl
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Laptop Price Calculator 🚀",
    page_icon="💻",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Add background color and title styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0px 4px 8px rgba(0,0,0,0.1);
    }
    .css-18e3th9 {
        padding-top: 2rem;
    }
    h1, h2 {
        text-align: center;
        color: #4CAF50;
    }
    h1 {
        font-size: 3rem;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 1rem;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# pipe = pkl.load(open("pipe.pkl", 'rb'))
# df = pkl.load(open("df.pkl", 'rb'))

# Title and Header
st.title("Laptop Price Calculator 💻")
st.subheader("Made with ❤️ by Ayush")

# # Load Model and Data
# pipe = pkl.load(open('C:\Users\ayush\Desktop\Laptop-Price-Calculator\pipe.pkl', 'rb'))
# df = pkl.load(open('C:\Users\ayush\Desktop\Laptop-Price-Calculator\df.pkl', 'rb'))
pipe = pkl.load(open('C:/Users/ayush/Desktop/Laptop-Price-Calculator/pipe.pkl', 'rb'))
df = pkl.load(open('C:/Users/ayush/Desktop/Laptop-Price-Calculator/df.pkl', 'rb'))


# import os
# pipe_path = os.path.join(os.path.dirname(__file__), "pipe.pkl")
# df_path = os.path.join(os.path.dirname(__file__), "df.pkl")

# pipe = pkl.load(open(pipe_path, "rb"))
# df = pkl.load(open(df_path, "rb"))




# Inputs
st.sidebar.header("Customize Laptop Specifications")
st.sidebar.markdown("Use the sidebar to select your preferences.")

company = st.sidebar.selectbox("💼 Brand", df['Company'].unique())
type = st.sidebar.selectbox("🖥️ Type of Laptop", df['TypeName'].unique())
ram = st.sidebar.selectbox("💾 RAM Size [in GB]", [2, 4, 6, 8, 16, 24, 32, 64])
weight = st.sidebar.number_input("⚖️ Weight of Laptop (in kg)", step=0.1)
touchscreen = st.sidebar.radio("💡 TouchScreen", ['Yes', 'No'])
ips = st.sidebar.radio("📺 IPS Display", ['Yes', 'No'])
screen_size = st.sidebar.number_input("📐 Screen Size (in inches)", step=0.1)
resolution = st.sidebar.selectbox(
    "🖼️ Resolution",
    ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440']
)
cpu = st.sidebar.selectbox("🧠 CPU", df['CpuBrand'].unique())
hdd = st.sidebar.selectbox("💽 HDD (in GB)", [0, 64, 128, 256, 512, 1024, 2048])
ssd = st.sidebar.selectbox("💾 SSD (in GB)", [0, 64, 128, 256, 512, 1024, 2048])
gpu = st.sidebar.selectbox("🎮 GPU", df['GpuBrand'].unique())
os = st.sidebar.selectbox("📀 Operating System", df['os'].unique())

# Predict Button
if st.sidebar.button("💸 Predict Price"):
    # Resolution processing
    x_res = int(resolution.split('x')[0])
    y_res = int(resolution.split('x')[1])
    ppi = ((x_res**2) + (y_res**2))**0.5 / screen_size

    # Convert categorical inputs to numerical
    touchscreen = 1 if touchscreen == "Yes" else 0
    ips = 1 if ips == "Yes" else 0

    # Create query for prediction
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    query = query.reshape(1, 12)

    # Predict price
    predicted_price = int(np.exp(pipe.predict(query))[0])

    # Display prediction
    st.success(f"💻 **Predicted Price**: ₹{predicted_price}")
    st.balloons()

# Add a footer
    st.markdown(
        """
        <hr>
        <footer style="text-align: center; color: #999;">
            Made with <span style="color: red;">❤️</span> using Streamlit | Ayush Budhlakoti
        </footer>
        """,
        unsafe_allow_html=True,
    )
    st.title('Predicted Price: ₹')
    st.title(int(np.exp(pipe.predict(query))[0]))
