#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Disable oneDNN optimizations (optional)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Configure the upload folder to store images
UPLOAD_FOLDER = 'static/uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained models (update with the correct paths)
main_model = load_model(r'C:\Users\SAI\best_model_xception_main.keras')  
stage_model = load_model(r'C:\Users\SAI\best_model_xception_stage.keras')  

# Class labels
main_classes = ['Not Infected', 'Infected']
stage_classes = ['ovale', 'malariae', 'vivax', ]

# Function to make predictions
def classify_image(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return model.predict(img_array)

# Function to calculate infection percentage using OpenCV
def calculate_infection_percentage(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    _, thresholded_image = cv2.threshold(blurred_image, 127, 255, cv2.THRESH_BINARY_INV)
    total_area = thresholded_image.size
    infected_area = np.sum(thresholded_image == 255)
    percentage_infected_area = (infected_area / total_area) * 100
    return percentage_infected_area

# Treatment recommendation function
def recommend_malaria_treatment(parasite_type):
    medications = {
        'falciparum': "ACTs are the recommended treatment. For severe cases, intravenous artesunate (2.4 mg/kg at 0, 12, and 24 hours) or quinine (10 mg/kg every 8 hours).",
        'vivax': "Chloroquine: 25 mg/kg over 3 days. Primaquine (0.25 mg/kg daily for 14 days) to prevent relapse.",
        'ovale': "Chloroquine: 20 mg/kg over 2 days. Primaquine (0.25 mg/kg daily for 14 days) for relapse prevention.",
        'malariae': "Chloroquine: 15 mg/kg over 3 days. No relapse prevention is necessary."
    }

    prevention = {
        'falciparum': ["Sleep under insecticide-treated nets.", "Use indoor residual spraying.", "Take antimalarial drugs like atovaquone-proguanil."],
        'vivax': ["Use mosquito repellents.", "Wear long-sleeved clothing.", "Consider prophylaxis with chloroquine or primaquine."],
        'ovale': ["Avoid outdoor activities during dusk and dawn.", "Use insecticide-treated nets.", "Prophylactic treatment with primaquine."],
        'malariae': ["Ensure indoor residual spraying.", "Take antimalarial prophylaxis.", "Eliminate mosquito breeding sites."]
    }

    treatment = medications.get(parasite_type, "Unknown parasite type")
    prevention_tips = prevention.get(parasite_type, ["Ensure proper diagnosis by a medical professional."])
    return treatment, prevention_tips

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file:
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(img_path)

        # Step 1: Classify if Infected or Not
        main_pred = classify_image(img_path, main_model)
        main_class_idx = np.argmax(main_pred, axis=1)[0]
        main_class = main_classes[main_class_idx]
        logging.info(f"Main class prediction: {main_class} (index: {main_class_idx})")

        # Initialize infection percentage
        infection_percentage = 0.0

        if main_class == 'Infected':
            # Step 2: Classify the type of parasite
            stage_pred = classify_image(img_path, stage_model)
            stage_class_idx = np.argmax(stage_pred, axis=1)[0]
            parasite_type = stage_classes[stage_class_idx]
            logging.info(f"Stage class prediction: {parasite_type} (index: {stage_class_idx})")

            # Step 3: Calculate the infection percentage
            infection_percentage = calculate_infection_percentage(img_path)
            logging.info(f"Infection percentage: {infection_percentage:.2f}%")

            # Step 4: Provide treatment recommendation
            treatment, prevention_tips = recommend_malaria_treatment(parasite_type)
            return render_template('result.html', 
                                   parasite_type=parasite_type, 
                                   treatment=treatment, 
                                   prevention_tips=prevention_tips, 
                                   img_path=img_path,
                                   infection_percentage=infection_percentage)
        else:
            return render_template('result.html', 
                                   parasite_type='Not Infected', 
                                   treatment='No treatment necessary.', 
                                   prevention_tips=[], 
                                   img_path=img_path,
                                   infection_percentage=infection_percentage)

if __name__ == '__main__':
    app.run(debug=True,port=5008,use_reloader=False)  # Set to True for development


# In[ ]:




