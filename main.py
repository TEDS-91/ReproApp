
# author: Tadeu Eder da Silva - Animal Scientist - PhD.
# University of Wisconsin - Madison

# imports required
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st
from PIL import Image
from skimage.transform import resize
from tensorflow.keras.models import load_model

st.title("Diagnóstico de Gestação App")

st.write("""
### Faça o upload da foto e obtenha o status de prenhez da vaca.
""")
# Disabling warning
st.set_option('deprecation.showfileUploaderEncoding', False)

# Choose your own image
uploaded_us = st.file_uploader(" ", type=['.JPG'])

image = image.load_img("C:/Users/tadeu/Downloads/0 - L5 - 34 (16196).jpg", target_size=(224, 224, 3))
show = st.image(image)

if uploaded_us is not None:
    u_img = Image.open(uploaded_us, 'r')
    u_img = u_img.convert('RGB')
    show.image(u_img, 'Uploaded ultrasound image')
    # We preprocess the image to fit in algorithm.
    image1 = np.asarray(u_img)
    my_image = resize(image1, (224, 224, 3), order=0, preserve_range=True, anti_aliasing=False)
else:
    st.error("Não há classificação sem imagem real. A imagem acima é apenas para ilustração.")
    st.stop()

img2 = np.expand_dims(my_image, axis=0)

saved_model = load_model("ResNet50FinalRepro.h5", compile=False)

output = saved_model(img2)

output = np.array(output)

if output[0][0] > 0.5:
    st.subheader('**Status**: Vaca **prenha** ({}% de probabilidade).'.format(round(output[0][0]*100,1)))
else:
    st.subheader('**Status**: Vaca **vazia** ({}% de probabilidade).'.format(round((1 - output[0][0])*100,1)))

expander = st.beta_expander("Dúvidas e questões:")
expander.write("Este trabalho é apenas um protótipo. Dúvidas e questões, favor contactar: **tdasilva2@wis.edu**")