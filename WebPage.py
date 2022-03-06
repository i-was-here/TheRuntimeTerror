from pyexpat import model
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tensorflow
from keras.models import model_from_json

json_file = open('model_in_json.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_weights.h5")
print("Loaded model from disk")

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#rec=cv2.face.LBPHFaceRecognizer_create()
# rec.read("trainingData.yml")

def main():
    """Face Recognition App"""

    st.title("THE RUNTIME TERROR")

    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Mask Detection</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        st.image(our_image)

    if st.button("Recognise"):
        our_image = tensorflow.image.rgb_to_grayscale(our_image)
        our_image = tensorflow.image.convert_image_dtype(our_image, tensorflow.float32)
        our_image = tensorflow.image.resize(our_image, (128,128))
        our_image = tensorflow.image.per_image_standardization(our_image)
        our_image = tensorflow.reshape(our_image, (1, 128, 128, 1))
        # result_img= loaded_model(our_image)
        result_img = loaded_model(our_image)
        if float(result_img[0][1])>float(result_img[0][0]) : st.text("Without Mask")
        else : st.text("With Mask")
        # st.image(result_img)


if __name__ == '__main__':
    main()
