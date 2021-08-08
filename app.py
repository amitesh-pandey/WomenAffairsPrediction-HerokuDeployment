from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)

filename = 'model2.pkl'
model = pickle.load(open(filename, 'rb'))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/submit", methods=['GET','POST'])
def predict():
    if request.method == "POST":
        intercept = request.form['Intercept']
        occ_2 = request.form['occ_2']
        occ_3 = request.form['occ_3']
        occ_4 = request.form['occ_4']
        occ_5 = request.form['occ_5']
        occ_6 = request.form['occ_6']
        occ_husb_2 = request.form['occ_husb_2']
        occ_husb_3 = request.form['occ_husb_3']
        occ_husb_4 = request.form['occ_husb_4']
        occ_husb_5 = request.form['occ_husb_5']
        occ_husb_6 = request.form['occ_husb_6']
        rate_marriage = request.form['rate_marriage']
        age = request.form['age']
        yrs_married = request.form['yrs_married']
        children = request.form['children']
        religious = request.form['religious']
        educ = request.form['educ']
        result = np.array([[intercept, occ_2, occ_3, occ_4, occ_5, occ_6,
                            occ_husb_2, occ_husb_3, occ_husb_4, occ_husb_5, occ_husb_6,
                            rate_marriage, age, yrs_married, children, religious, educ]])
        prediction = model.predict(result)

    return render_template("submit.html", n = prediction)




if __name__ =="__main__":
    app.run(debug = True)


