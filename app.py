from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)
# Flask(__name__) it will gives us the entry point where we need to execute it
# Flask(__name__) tells Flask where the application is located so it can find templates, static files, and resources correctly.
# Flask(__name__) = Location reference for your Flask app

app = application
## Route for home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')   # data fields where user will enter data
    else:  # POST .. Where user will select different values and using this we get the input from user and try to predict the output.
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity= request.form.get('race_ethnicity'),
            parental_level_of_education= request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score= float(request.form.get('reading_score')),
            writing_score= float(request.form.get('writing_score')),
        )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()

        results = predict_pipeline.predict(pred_df)

        return render_template('home.html',results = results[0])

if __name__ == "__main__":
    app.run(host= "0.0.0.0",debug = True)

'''
Render template: It connects your backend logic with your frontend page.
Load home.html and replace {{ results }} with the value of prediction.
What does host mean?
The host defines which network address your Flask app will listen on.
It tells Flask: “From where should people be able to access my app?”
'''