import streamlit as st
from PIL import Image
import prediction as pr


st.title("Sunglasses Lens Color Image Classification App")
st.write("")


file_up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

image = Image.open(file_up)
st.image(image, caption=pr.get_prediction(image), channels='RGB', use_column_width=True)

