from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

import tensorflow as tf
import os
import matplotlib.pyplot as plt
from numpy import *
from tensorflow.keras.models import load_model
import tensorflow_hub as hub

import numpy as np


import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

app = Flask('__name__')


file_path_g=""


# Load the models with the custom_objects parameter
custom_objects = {
    'KerasLayer': hub.KerasLayer
}

model = load_model('diabetic.keras', custom_objects=custom_objects)


@app.route('/', methods = ['GET', 'POST'])
def intro_start():
    return render_template("welcome.html")


import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble



import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble


def create_quantum_circuit(image_features):
    num_features = len(image_features)
    quantum_register = QuantumCircuit(num_features, num_features)  # Define quantum and classical registers
    
    # Apply quantum operations to encode image features into quantum states
    for i in range(num_features):
        # Convert array to scalar (taking the mean of pixel values for simplicity)
        value = np.mean(image_features[i])
        
        # Scale the value to a suitable range for quantum gates (e.g., [0, pi])
        scaled_value = value * np.pi / 255.0
        
        # Apply rotation gate based on scaled feature value
        quantum_register.ry(scaled_value, i)
        
    quantum_register.measure_all()
        
    return quantum_register


def qsdl(set1):
    train_q = set1
    #train_q = set1

    # Create an array to store the measurement outcomes
    num_images = len(train_q)  
    num_qubits = 6  # Number of qubits
    num_outcomes = 2 ** num_qubits  # Possible outcomes for 6 qubits

    quantum_data = np.zeros((num_images, num_outcomes), dtype=int)

    # Simulate each quantum circuit and store measurement outcomes
    simulator = Aer.get_backend('aer_simulator')
    for i, image_features in enumerate(train_q):
        qc = create_quantum_circuit(image_features)
        compiled_circuit = transpile(qc, simulator)
        qobj = assemble(compiled_circuit)
        result = simulator.run(qobj).result()
        counts = result.get_counts()


        for outcome, count in counts.items():
            original_string = outcome
            without_spaces = "".join(original_string.split())
            outcome_int = int(without_spaces, 2)

            if outcome_int < num_outcomes:
                quantum_data[i, outcome_int] = count

    # Print the quantum data array
    print("Quantum data created")
    return quantum_data


def pred(name):
    
    #list to hold image 
    unseen=[]
    
    display_image=plt.imread(name)
    
    #load the image
    image=cv2.imread(name)

    #resize the image
    image=cv2.resize(image, (64,64))

    #append the image
    unseen.append(image)

    #converting and normalizing
    unseen_images = np.array(unseen)
    unseen_images = unseen_images.astype('float32') / 255.0

    #call for creating quantum_data for unseen_test_data
    quantum_unseen=qsdl(unseen_images[:1])
    
    
    # Make predictions using the model
    predictions = model.predict([unseen_images[:1], quantum_unseen[:1]])

    # Print the predicted probabilities for each class
    print("Predicted probabilities:")
    #print(predictions)

    # Find the index of the class with the highest predicted probability for each example
    predict_label = np.argmax(predictions, axis=1)

    # Find the maximum predicted label value
    max_predicted_label = np.max(predict_label)

    print("Predicted labels:", predict_label)
    #print("Maximum predicted label:", max_predicted_label)

    Actual={0:"Mild",1:"Moderate",2:"No_DR",3:"Proliferate_DR",4:"Severe"}
    print("The class of the images is-->", Actual[max_predicted_label])

    return Actual[max_predicted_label]



@app.route('/detect', methods = ['GET', 'POST'])
def upload_detection():
    
    if(request.method == 'POST'):
        f=request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)


        global file_path_g

        file_path_g=file_path


        
    return render_template("detect.html")




@app.route('/result', methods = ['GET', 'POST'])
def extract_result():

    # Make prediction
       
    val=pred(file_path_g)

    return render_template("close.html",n="Your Retina Is At "+str(val)+" Severity")


    
if __name__ == "__main__":
    app.run(debug = False)
