import numpy as np
import pickle

from flask import Flask,render_template,url_for,request

app=Flask(__name__)
model = pickle.load(open('heart_attack.pkl', 'rb'))
model1 = pickle.load(open('cancer.pkl','rb'))
model2= pickle.load(open('chronic-kidney.pkl','rb'))
model3=pickle.load(open('stroke.pkl','rb'))
model4=pickle.load(open('hepatitis.pkl','rb'))
model5=pickle.load(open('diabetes.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


#---------homes------

@app.route('/heart')
def heart_home():
    return render_template('heart_home.html')

@app.route('/breast_cancer')
def breast_cancer_home():
    return render_template('breast_cancer_home.html')

@app.route('/chronic_kidney')
def chronic_kidney_home():
    return render_template('chronic_kidney_home.html')

@app.route('/stroke')
def stroke_home():
    return render_template('stroke_home.html')

@app.route('/hepatitis')
def hepatitis_home():
    return render_template('hepatitis_home.html')


@app.route('/diabetes')
def diabetes_home():
    return render_template('diabetes_home.html')   
#------------heart-------
@app.route("/heart_predict", methods=['POST'])
def heart_predict():
    if request.method == 'POST':
        chest_pain_type = request.form['chest_pain_type']
        Exercise_Include_Angina = request.form['Exercise_Include_Angina']
        oldpeak = request.form['oldpeak']
        caa = request.form['caa']
        thall = request.form['thall']
        values = np.array([[chest_pain_type,Exercise_Include_Angina,oldpeak,caa,thall]])
        prediction = model.predict(values)

        return render_template('heart_result.html', prediction=prediction)

#-------breast-cancer----------


@app.route('/breast_cancer_predict',methods=['POST'])
def breast_cancer_predict():
     if request.method == 'POST':
        MeanRadius = float(request.form['MeanRadius'])
        MeanTexture = float(request.form['MeanTexture'])
        MeanPerimeter = float(request.form['MeanPerimeter'])
        MeanArea = float(request.form['MeanArea'])
        MeanSmoothing = float(request.form['MeanSmoothness'])

        values = np.array([[MeanRadius,MeanTexture,MeanPerimeter,MeanArea,MeanSmoothing]])
        prediction = model1.predict(values)

        return render_template('breast_cancer_result.html', prediction=prediction)


@app.route('/chronic_kidney_predict',methods=["POST"])
def chronic_kidney_predict():
    if request.method=="POST":
        specific_gravity=float(request.form["specific_gravity"])
        serum_creatinine=float(request.form["serum_creatinine"])
        hemoglobin=float(request.form["hemoglobin"])
        packed_cell_volume=float(request.form["packed_cell_volume"])
        red_blood_cell_count=float(request.form["red_blood_cell_count"])

    values=np.array([[specific_gravity,serum_creatinine,hemoglobin,packed_cell_volume,red_blood_cell_count]])
    prediction=model2.predict(values)

    return render_template('chronic_kidney_result.html',prediction=prediction)


@app.route('/stroke_risk_predict',methods=["POST"])
def stroke_risk_predict():
    if request.method=="POST":
        age=request.form["age"]
        ever_married=request.form["ever_married"]
        work_type=request.form["work_type"]
        caa=request.form["caa"]
        bmi=request.form["bmi"]

    values=np.array([[age,ever_married,work_type,caa,bmi]])
    prediction=model3.predict(values)

    return render_template('stroke_result.html',prediction=prediction)


@app.route('/hepatitis_predict',methods=["POST"])
def hepatitis_predict():
    if request.method=="POST":
        age=request.form["age"]
        bilirubin=request.form["bilirubin"]
        alk_phosphate=request.form["alk_phosphate"]
        sgot=request.form["sgot"]
        albumin=request.form["albumin"]

    values=np.array([[age,bilirubin,alk_phosphate,sgot,albumin]])
    prediction=model4.predict(values)

    return render_template('hepatitis_result.html',prediction=prediction)



@app.route('/diabetes_predict',methods=["POST"])
def diabetes_predict():
    if request.method=="POST":
        Pregnancies=request.form["pregnancies"]
        Glucose=request.form["glucose"]
        BloodPressure=request.form["blood_pressure"]
        SkinThickness=request.form["skin_thickness"]
        Insulin=request.form["insulin"]
        BMI=request.form["bmi"]
        DiabetesPedigreeFunction=request.form["diabetes_pedigree_function"]
        Age=request.form["age"]

    values=np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    prediction=model5.predict(values)

    return render_template('diabetes_result.html',prediction=prediction)


@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True)