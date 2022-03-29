import streamlit as st
from PIL import Image, ImageDraw,ExifTags,ImageColor
import os
import boto3

st.header('Animal Detection on AWS')

def load_image(image_file):
    img=Image.open(image_file)
    return img

col1,col2=st.columns(2)
src_image_file=col1.file_uploader("Upload Image",type=["png","jpg","jpeg"],key=1)

col3,col4=st.columns(2)
col3.text('Person')
col3.text('Tiger')
col3.text('Lion')
col3.text('Deer')

person_bar = col4.progress(0)
tiger_bar  = col4.progress(0)
lion_bar   = col4.progress(0)
deer_bar   = col4.progress(0)
if src_image_file is not None:
    file_details={"filename":src_image_file,"filetype":src_image_file.type,"filesize":src_image_file.size}
    #st.write(file_details)
    #st.image(load_image(src_image_file),width=250)

    with open(os.path.join("uploads","src.jpg"),"wb") as f:
        f.write(src_image_file.getbuffer())

col2.text(' ')
col2.text(' ')
col2.text(' ')
col2.text(' ')


if col2.button('Predict'):
    imageSource=open("uploads/src.jpg",'rb')
    client=boto3.client('rekognition')
    response=client.detect_labels(Image={'Bytes':imageSource.read()},MaxLabels=1)
    label=(response['Labels'][0]['Name'])
    st.success(label)
    
    if(label=='Face'):
        person_bar.progress(response['Labels'][0]['Confidence']/100)
    elif(label=='Tiger'):
        tiger_bar.progress(response['Labels'][0]['Confidence']/100)
    elif(label=='Lion'):
        lion_bar.progress(response['Labels'][0]['Confidence']/100)
    elif(label=='Deer'):
        deer_bar.progress(response['Labels'][0]['Confidence']/100)


    


    