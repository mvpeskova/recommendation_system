# Project description
This project involves analyzing data from the file 'insurance.csv', which contains details about medical insurance costs in the USA based on various factors: age, gender, Body Mass Index (BMI), number of children, and smoking status. The recommendation system is developed to provide an estimated cost of medical insurance for a user, based on their input parameters (which can be changed in the GUI). The general trend in insurance costs is also reflected through scatter and stem diagrams.

# Prerequisites

Python 3.8.10

PyQt 6.6.1

Pandas 2.0.3

Numpy 1.24.4

Matplotlib 3.7.4

Scikit-Learn 1.3.2

seaborn

# Installation
 
pip install -r requirements.txt

# Basic Usage

Launch the application. You will see the GUI with input fields: age, sex, BMI, number of Children, smoker, and region.
Adjust these parameters according to the individual's profile for whom you want to predict the insurance cost.
Select either 'Scatter' or 'Stem' from the Visualization dropmenu to view analyzed data in different forms.
After setting the input parameters, click on the 'PREDICT' button.
The system will then display the recommended insurance charges based on the input parameters.

The application provides statistics metrics: mean, mode, and standard deviation for BMI, Children, and Age. You can see them under the diagram.

