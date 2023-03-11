import streamlit as st
from PIL import Image
st.set_page_config(
    page_title="#3D model using Stereo Images",
    page_icon="👋",
)

st.write("# Convert a stereo image into a 3D model!!! 👋")

st.sidebar.success("Select from above.")

st.write("This webapp provides you with a way to convert your images to a 3D model. ")

st.write("You just require a stereo image!!")

st.header("What's a Stereo Image?")

st.write("Stereo Images or Stereoscopic pictures are produced in pairs, the members of a pair showing the same scene or object from slightly different angles that correspond to the angles of vision of the two eyes of a person looking at the object itself.")

image = Image.open('images.jpg')

st.image(image)

st.header("Access the 3D model maker and 3D model processor in the side bar! ")