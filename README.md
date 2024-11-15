# Maleria-parasites-detection-and-classification
Project: Malaria Classification and Stage Detection Using Xception<br>
Description:<br>
This project leverages the Xception deep learning architecture to classify malaria infection and determine its stage.<br> It involves building two models:<br>

Main Classification Model:<br>

Identifies whether a blood smear image indicates "Infected" or "Not Infected."<br>
Uses transfer learning with a pre-trained Xception model for enhanced accuracy.<br>
Stage Classification Model:<br>

Determines the specific stage of malaria infection from infected images.<br>
Handles multiple class predictions for stage identification.<br>
Key Features:<br>
Dataset Handling: Processes a dataset with subdirectories for "Infected" and "Not Infected" classes, and further subcategories under "Infected" for stage classification.<br>
Transfer Learning: Utilizes Xception with pre-trained weights (imagenet) for both models, freezing base layers for efficient transfer learning.<br>
Data Augmentation: Applies transformations (e.g., rotation, zoom, shift, shear, brightness, horizontal flip) to improve model generalization.<br>
Custom Architectures: Adds dense and dropout layers to the Xception backbone for both tasks.<br>
Callbacks: Includes early stopping, model checkpoints, and learning rate reduction for efficient training.<br>
Performance Evaluation: Evaluates models on validation datasets with accuracy and loss metrics.<br>
Visualization:<br>
Displays prediction results with matplotlib, highlighting correct and incorrect predictions.<br>
Provides insights into model performance with clear visual representations.<br>
Technologies Used:<br>
Python, TensorFlow, Keras<br>
Xception architecture for transfer learning<br>
Data Augmentation with ImageDataGenerator<br>
Model deployment-ready with saved .keras models<br>
Achievements:<br>
Implemented robust classification with high accuracy on the validation dataset.<br>
Integrated a modular and reusable architecture for similar image classification tasks.<br>
Demonstrated effective data visualization for better interpretation of predictions.<br>
This project demonstrates expertise in deep learning, image classification, and transfer learning techniques.<br> It is a practical solution for automating malaria diagnosis and staging in healthcare applications.<br>
