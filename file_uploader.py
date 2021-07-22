import streamlit as st
from PIL import Image
import prediction as pr

file_up = st.file_uploader("Upload an image", type="jpg")

image = Image.open(file_up)
st.image(image, caption=pr.get_prediction(image), channels='RGB', use_column_width=True)

