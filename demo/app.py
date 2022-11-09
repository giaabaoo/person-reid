import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
import torch
from demo.main import *
plt.rcParams["figure.figsize"] = (10, 7)
from PIL import Image
from helper import predict_results

def load_image(image_file):
	img = Image.open(image_file).convert("RGB")
	return img


def action(image, text):
    if image is not None:
        # To See details
        file_details = {"filename":image.name, "filetype":image.type,
                        "filesize":image.size}
        st.write(file_details)
        st.image(load_image(image), caption="Uploaded Image", use_column_width=True)
        
    predictions = retrieve_results(image, text)
    st.image(predictions, caption="Retrieved Images ranked from left to right", use_column_width=True)
    st.success("Here are the relevant results according to your query!")
    st.balloons()


def retrieve_results(image, text):
    predictions = predict_results(image, text)

    return predictions


def main():
    placeholder1 = st.empty()
    placeholder2 = st.empty()
    st.markdown(
        "# Person Reidentification App\n"
        "Upload a person image with a text description\n\n. "
    )

    if True:
        col1, col2, col3 = st.columns([1,9,1])

        with col1:
            st.write("")

        with col2:
            st.image("demo/demo.gif")

        with col3:
            st.write("")

        st.subheader("Author:")
        members = ''' 
        Gia-Bao Dinh Ho\n
        '''
        st.markdown(members)

        st.success("Gia-Bao Dinh Ho")
    # st.sidebar.image("demo/assets/demoo.gif")

    st.sidebar.markdown("Upload image:")
    file_uploader = st.sidebar.file_uploader(
        label="", type=[".jpg", ".png", ".jpeg"]
    )
    user_input = st.text_input("And the description for the image: ", "The")

    st.sidebar.markdown("---")
    if st.sidebar.button("Retrieve the person from other views"): 
        placeholder1.empty()
        placeholder2.empty()

        action(
            image=file_uploader,
            text=user_input
        )


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Person re-identificaiton with text descriptions")
    main()