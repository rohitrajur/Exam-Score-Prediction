import pickle
from flask import Flask, render_template, request
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model (make sure model.pkl is in the same directory)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Data for dropdowns based on the category counts you provided
dropdown_data = {
    'school_setting': ['Urban', 'Suburban', 'Rural'],
    'school_type': ['Public', 'Non-public'],
    'teaching_method': ['Standard', 'Experimental'],
    'lunch': ['Does not qualify', 'Qualifies for reduced/free lunch']
}

# Route for the homepage (HTML form)
@app.route('/')
def home():
    return render_template('index.html', dropdown_data=dropdown_data)

# Route to handle form submission and make prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Fetch data from form inputs
        school_setting = request.form['school_setting']
        school_type = request.form['school_type']
        teaching_method = request.form['teaching_method']
        n_student = int(request.form['n_student'])
        lunch = request.form['lunch']
        pretest = float(request.form['pretest'])

        # Convert categorical variables to appropriate numerical values
        # Example: You can use a simple mapping if your model needs numerical values
        school_setting_map = {'Urban': 0, 'Suburban': 1, 'Rural': 2}
        school_type_map = {'Public': 0, 'Non-public': 1}
        teaching_method_map = {'Standard': 1, 'Experimental': 0}
        lunch_map = {'Does not qualify': 0, 'Qualifies for reduced/free lunch': 1}

        # Prepare the features for prediction
        features = np.array([
            school_setting_map.get(school_setting, 0),
            school_type_map.get(school_type, 0),
            teaching_method_map.get(teaching_method, 0),
            n_student,
            lunch_map.get(lunch, 0),
            pretest
        ]).reshape(1, -1)

        # Predict the outcome using the trained model
        prediction = model.predict(features)

        # Render the result on a new page
        return render_template('index.html', prediction=prediction[0], dropdown_data=dropdown_data)

if __name__ == '__main__':
    app.run(debug=True)
