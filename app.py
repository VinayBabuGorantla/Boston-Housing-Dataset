
# app.py

from flask import Flask, request, render_template, jsonify
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                crim=float(request.form['crim']),
                zn=float(request.form['zn']),
                indus=float(request.form['indus']),
                chas=float(request.form['chas']),
                nox=float(request.form['nox']),
                rm=float(request.form['rm']),
                age=float(request.form['age']),
                dis=float(request.form['dis']),
                rad=float(request.form['rad']),
                tax=float(request.form['tax']),
                ptratio=float(request.form['ptratio']),
                b=float(request.form['b']),
                lstat=float(request.form['lstat'])
            )

            final_df = data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            prediction = predict_pipeline.predict(final_df)

            logging.info(f"Prediction made: {prediction[0]}")
            return render_template('home.html', results=prediction[0])
        
        except Exception as e:
            logging.error(f"Error occurred: {e}")
            return render_template('home.html', results="Error occurred")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
