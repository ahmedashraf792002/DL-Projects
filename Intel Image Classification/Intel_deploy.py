import streamlit as st
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np

# Set page icon
st.set_page_config(page_icon=":smiley:")
st.markdown(
    """
    <style>
    * {
        color: #fca311;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Load pre-trained machine learning model
model = tf.keras.models.load_model('D:/Project/Python/project/model_transfer.h5')
code={'buildings':0 ,'forest':1,'glacier':2,'mountain':3,'sea':4,'street':5}
def get_Name(N):
    for x,y in code.items():
          if y==N:
                return x

# Create a function to make predictions on an image
def make_prediction(image):
    # Preprocess image here
    image = cv2.resize(image, (150, 150))
    image = np.expand_dims(image, axis=0)#reshape(1,100,100,3)
    prediction = model.predict(image)#prediction
    for i in range(6):
        st.write('Prediction:',f'{get_Name(i)}\t\tPrecentage  {round(prediction[0][i])*100}%')
    st.write('------------------------') 
    N=np.argmax(prediction)
    st.write('Label is:',f'{get_Name(N)}')

# Create a Streamlit app
st.title("Intel Image Classification App")
# Create a button to upload an image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Load image from file
    image = Image.open(uploaded_file)
    # Convert PIL Image to numpy array
    # Display image in Streamlit app
    st.image(image, caption='My Image', use_column_width=True)
else:
    st.write('Please upload an image first')

# Create a button to make a prediction
if st.button('Make Prediction'):
    img = np.array(image)
    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Make prediction on image
    make_prediction(img)
# Create a button to clear the uploaded image and prediction
if st.button('Clear'):
    uploaded_file = None
    #st.image([], use_column_width=True)
    st.write('Upload and prediction cleared')