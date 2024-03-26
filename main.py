import pandas as pd
import numpy as np
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QMessageBox, QCheckBox, 
                             QLabel, QPushButton, QSlider, QHBoxLayout, QVBoxLayout, 
                             QSpinBox, QRadioButton, QCheckBox, QComboBox)

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures

# Read insurance data from the CSV file
insurance_data = pd.read_csv('insurance.csv')
analysis_data = insurance_data

# Data overview
print(f"{insurance_data.head()}\n")
print(f"{insurance_data.describe()}\n")
print(f"{insurance_data.info()}\n")

# For saving graphs
figure_id = 0

insurance_data = pd.get_dummies(insurance_data, columns=['sex', 'smoker', 'region'], drop_first=True)

# Calculate the correlation with charges
correlation_with_charges = insurance_data.corr()['charges'].sort_values(ascending=False)
print(f"Correlation with charges: \n{correlation_with_charges}")


# Main window class
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        widget = AppWidget()
        self.setCentralWidget(widget)
        self.setWindowTitle("Health Insurance Costs Recommendation")
        self.setFixedSize(1000, 700)
    
    def closeEvent(self, event):
        reply = QMessageBox.question(self, "Message", "Do you want to close this application?",
            QMessageBox.StandardButton.Close | QMessageBox.StandardButton.Cancel)

        if (reply == QMessageBox.StandardButton.Close):
            event.accept()
            sys.exit()
        else:
            event.ignore()


# Main widget class
class AppWidget(QWidget):
    def __init__(self):
        super().__init__()
        # Initialize the linear regression model
        self.predicted_charge = 0
        self.X_train, self.X_test = None, None
        self.regr = linear_model.LinearRegression()
        self.initModel(analysis_data)

        # Initializing GUI elements
        font = QFont()
        font.setPointSize(11)

        font1 = QFont()
        font1.setPointSize(11)

        self.method_label = QLabel("Visualization:")
        self.method_combobox = QComboBox()
        self.method_combobox.addItems(["Scatter", "Stem"])

        self.age_label = QLabel("Age:")
        self.age_spinbox = QSpinBox()
        self.age_spinbox.setValue(20)

        self.sex_label = QLabel("Sex:")
        self.sex_male = QRadioButton("Male")
        self.sex_female = QRadioButton("Female")
        self.sex_male.setChecked(True)

        self.children_slider = QSlider(Qt.Orientation.Horizontal)
        self.children_slider.setMinimum(0)
        self.children_slider.setMaximum(10)
        self.children_value_label = QLabel("Children: " + str(self.children_slider.value()), self)
        self.children_slider.valueChanged.connect(self.update_children_label)

        self.bmi_label = QLabel("BMI:")
        self.bmi_spinbox = QSpinBox()
        self.bmi_spinbox.setValue(20)

        self.smoker_checkbox = QCheckBox("Smoker")

        self.region_label = QLabel("Region:")
        self.region_combobox = QComboBox()
        self.region_combobox.addItems(["Northwest", "Northeast", "Southwest", "Southeast"])

        self.submit_button = QPushButton("PREDICT")

        self.save_button = QPushButton("SAVE GRAPH")

        self.result_label = QLabel("Recommended charges: \n\nAge: \nSex: \nChildren: \nSmoker: \nBMI: \nRegion: ")

        self.result_label.setFont(font)
        # Vertical layout
        self.layout = QVBoxLayout()

        # Calculate statistic measures
        bmi_mean = insurance_data['bmi'].mean()
        bmi_mode = insurance_data['bmi'].mode()
        bmi_std = insurance_data['bmi'].std()

        children_mean = insurance_data['children'].mean()
        children_std = insurance_data['children'].std()
        children_mode = insurance_data['children'].mode()

        age_mean = insurance_data['age'].mean()
        age_std = insurance_data['age'].std()
        age_mode = insurance_data['age'].mode()
      
        self.bmi_stat_label = QLabel(f"BMI:\nMean: {bmi_mean:.2f}\nMode: {bmi_mode.values[0]}\nStandard Deviation: {bmi_std:.2f}", self)
        self.bmi_stat_label.setFont(font)

        self.children_stat_label = QLabel(f"Children:\nMean: {children_mean:.2f}\nMode: {children_mode.values[0]}\nStandard Deviation: {children_std:.2f}", self)
        self.children_stat_label.setFont(font)

        self.age_stat_label = QLabel(f"Age:\nMean: {age_mean:.2f}\nMode: {age_mode.values[0]}\nStandard Deviation: {age_std:.2f}", self)    
        self.age_stat_label.setFont(font)

        # Horizontal layout
        self.stat_layout = QHBoxLayout()

        # Add widgets to the horizontal layout
        self.stat_layout.addWidget(self.bmi_stat_label)
        self.stat_layout.addWidget(self.age_stat_label)
        self.stat_layout.addWidget(self.children_stat_label)

        self.layout.addWidget(self.method_label)
        self.layout.addWidget(self.method_combobox)

        self.layout.addWidget(self.save_button)
        self.save_button.clicked.connect(self.save_graph)

        self.layout.addWidget(self.age_label)
        self.layout.addWidget(self.age_spinbox)
        self.layout.addSpacing(20)

        self.layout.addWidget(self.sex_label)
        self.layout.addWidget(self.sex_male)
        self.layout.addWidget(self.sex_female)
        self.layout.addSpacing(20)

        self.layout.addWidget(self.children_slider)
        self.layout.addWidget(self.children_value_label)
        self.layout.addSpacing(20)

        self.layout.addWidget(self.bmi_label)
        self.layout.addWidget(self.bmi_spinbox)
        self.layout.addSpacing(20)

        self.layout.addWidget(self.smoker_checkbox)
        self.layout.addSpacing(20)

        self.layout.addWidget(self.region_label)
        self.layout.addWidget(self.region_combobox)
        self.layout.addSpacing(20)

        self.layout.addWidget(self.submit_button)
        self.layout.addWidget(self.result_label)

        self.submit_button.clicked.connect(self.display_result)

        self.canvas = FigureCanvasQTAgg(Figure(figsize=(7, 6)))

        # Vertical layout
        self.canvas_layout = QVBoxLayout()
        self.canvas_layout.addWidget(self.canvas)
        self.canvas_layout.addLayout(self.stat_layout)

        # Main layout
        self.main_layout = QHBoxLayout()
        # Add both layouts to the main layout
        self.main_layout.addLayout(self.layout)
        self.main_layout.addLayout(self.canvas_layout)

        self.setLayout(self.main_layout)
        self.plot_data(20.0, 0)

    def update_children_label(self):
        self.children_value_label.setText("Children: " + str(self.children_slider.value()))

    def display_result(self):
        # Retrieve values from the GUI elements
        age = float(self.age_spinbox.value())
        bmi = float(self.bmi_spinbox.value())
        children = self.children_slider.value()

        if self.sex_male.isChecked():
            sex = "Male"
        else:
            sex = "Female"

        if self.smoker_checkbox.isChecked():
            smoker = "Yes"
        else:
            smoker = "No"

        region = self.region_combobox.currentText()

        transformed_input = self.transform_input(age, sex, bmi, children, smoker, region)

        transformed_input_df = pd.DataFrame([transformed_input], columns=self.X_test.columns)

        # Make prediction
        self.predicted_charge = self.regr.predict(transformed_input_df)[0]

        self.plot_data(bmi, self.predicted_charge)

        result = f"Recommended charges: ${round(self.predicted_charge, 2)}\n\nAge: {age} \nSex: {sex} \nChildren: {children}\nSmoker: {smoker} \nBMI: {bmi} \nRegion: {region}"
        self.result_label.setText(result)

    def transform_input(self, age, sex, bmi, children, smoker, region):
        input_data = {col: 0 for col in self.X_test.columns}

        input_data['age'] = age
        input_data['bmi'] = bmi
        input_data['children'] = children

        if sex == "Male":
            input_data['sex_male'] = 1

        if smoker == "Yes":
            input_data['smoker_yes'] = 1

        region_map = {
            "Northwest": 'region_northwest',
            "Northeast": 'region_northeast',
            "Southwest": 'region_southwest',
            "Southeast": 'region_southeast'
        }
        if region in region_map:
            input_data[region_map[region]] = 1

        # Return the values in the same order as in the training data
        return [input_data[col] for col in self.X_test.columns]

    def initModel(self, analysis_data):
        # Initialize the linear regression model using the training data
        categorical_cols = ['sex', 'smoker', 'region']
        analysis_data = pd.get_dummies(analysis_data, columns=categorical_cols)

        X = analysis_data.drop('charges', axis=1)
        Y = analysis_data["charges"]

        self.X_train, self.X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        self.regr.fit(self.X_train, y_train)
        Y_pred = self.regr.predict(self.X_test)
        print("Coefficients: \n", self.regr.coef_)
        # The mean squared error
        print("Mean squared error: %.2f" % mean_squared_error(y_test, Y_pred))
        print("Coefficient of determination: %.2f" % r2_score(y_test, Y_pred))

    def plot_data(self, user_bmi, predicted_charge):
        # Plot the data using matplotlib and update the canvas
        self.canvas.figure.clf()
        ax = self.canvas.figure.add_subplot(111)

        colour = '#67359c'

        if self.method_combobox.currentText() == "Scatter":

            ax.scatter(analysis_data['bmi'], analysis_data['charges'], color="#9fa6f5")
            ax.scatter(user_bmi, predicted_charge, color='#67359c', marker='*', s=100)
        else:
            ax.stem(analysis_data['bmi'], analysis_data['charges'], linefmt='#fc9fcb', markerfmt='o', basefmt='r-')
            colour = "#c74472"
            ax.stem([user_bmi], [predicted_charge], linefmt='#c74472', markerfmt='*', basefmt='r-')

        bmi_values = np.linspace(analysis_data['bmi'].min(), analysis_data['bmi'].max(), 100)
        df_for_plot = pd.DataFrame(np.zeros((100, len(self.X_train.columns))), columns=self.X_train.columns)
        df_for_plot['bmi'] = bmi_values
        charges_predicted = self.regr.predict(df_for_plot)
        ax.plot(bmi_values, charges_predicted, color=colour)

        ax.set_title('Charges Prediction')
        ax.set_xlabel('BMI')
        ax.set_ylabel('Charges')
        self.canvas.draw()

    def save_graph(self):
        # Save the current graph
        global figure_id
        image_name = f"graph{figure_id}"
        self.canvas.figure.savefig(image_name)
        figure_id += 1
        print("Graph saved")

if __name__ == '__main__':
    app = QApplication([])
    main_app = MainWindow()
    main_app.show()
    app.exec()
