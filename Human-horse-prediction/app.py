#Import necessary libraries
from flask import Flask, render_template, request

import numpy as np
import os

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
 

def pred_human_horse(model , horse_or_human):
  test_image = load_img(horse_or_human , target_size=(150,150)) #resize
  print("@@ Got Image for predicton")
  test_image = img_to_array(test_image)/255 #numpy array  between 0-1
  test_image = np.expand_dims(test_image,axis=0) #4 dimension

  result= model.predict(test_image).round(3) #rounding off
  pred =np.argmax(result)
  print("@@ Raw results = ",result)
  print("@@ class = ", pred)

  if pred==0:
    return "Horse"

  else:
    return "Human"

# Crate flask app
app = Flask(__name__)
    
@app.route("/",methods=["GET","POST"])
def home():
    return render_template("index.html")


@app.route("/predict",methods=["GET","POST"])
def predict():
    if request.method=="POST":
        #get input image file
        file = request.files["image"]
        filename= file.filename
        print("@@ File recieved",filename)
        
        #save the file
        file_path= os.path.join("static/user_uploaded",filename)
        file.save(file_path)
        
        print("@@ Prediction...")
        #load model
        model = load_model("predictor.h5")
        print("@@ model loaded")

        pred=pred_human_horse(model,horse_or_human=file_path )
        
        return render_template("predict.html" ,pred_output= pred , user_image=file_path )


if __name__=="__main__":
    app.run(threaded=False)