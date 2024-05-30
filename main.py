# -*- coding: utf-8 -*-
"""
Created on Thu May 30 21:49:38 2024

@author: yadhu
"""

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

working_dir=os.path.dirname(os.path.abspath(__file__))
model_path=f"{working_dir}/trained_fashion_mnist_model.h5"
model=tf.keras.models.load_model(model_path)
class_names=['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']

def preprocess_image(image):
    img=Image.open(image)
    img=img.resize((28,28))
    img=img.convert('L')
    img_array=np.array(img)/255
    img_array=img_array.reshape((1,28,28,1))
    return img_array

st.title('Fashion Item Classifier')
uploaded_image=st.file_uploader("upload an image",type=['jpg','png','jpeg'])

if(uploaded_image is not None):
    image=Image.open(uploaded_image)
    col1,col2=st.columns(2)
    
    with col1:
        resized_image=image.resize((100,100))
        st.image(resized_image)
        
    with col2:
        if(st.button('classify')):
            img_array=preprocess_image(uploaded_image)
            result=model.predict(img_array)
            predicted_class=np.argmax(result)
            prediction=class_names[predicted_class]
            
            st.success(prediction)
    
    