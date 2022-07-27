# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 12:16:47 2022

@author: Sajidha
"""

import requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask,request,render_template,redirect,url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session
app = Flask(__name__)
model=load_model(r"C:\Users\Sajidha\fruit.h5")
model1=load_model(r"C:\Users\Sajidha\veg.h5")
#homepage
@app.route('/')
def home():
    return render_template('home.html')
#prediction page
@app.route('/prediction')
def prediction():
    return render_template('predict.html')
@app.route('/predict',methods=['POST'])
def predict():
    if request.method =='POST':
       f=request.files['image']
       basepath=os.path.dirname(__file__)
       file_path=os.path.join(basepath,'uploads',secure_filename(f.filename))
       f.save(file_path)
       img=image.load_img(file_path,target_size=(128,128))
       x=image.img_to_array(img)
       x=np.expand_dims(x,axis=0)
       plant=request.form['plant']
       print(plant)
       if(plant=="vegetable"):
           preds= np.argmax(model.predict(x))
           print(preds)
           df=pd.read_excel(r'D:\VIT_PI_21_22\WIN_21_22\IBM_BADGE_COURSE\Fertilizers Recommendation System For Disease Prediction_2\Fertilizers Recommendation System For Disease Prediction\precautions - fruits.xlsx')
           print(df.iloc[preds]['caution'])
       else:
           preds=np.argmax(model1.predict(x))
           df=pd.read_excel(r'D:\VIT_PI_21_22\WIN_21_22\IBM_BADGE_COURSE\Fertilizers Recommendation System For Disease Prediction_2\Fertilizers Recommendation System For Disease Prediction\precautions - veg.xlsx')
           print(df.iloc[preds]['caution'])
    return df.iloc[preds]['caution']    
if __name__=="__main__":
   app.run(debug=False)
       
           
       
       
       




