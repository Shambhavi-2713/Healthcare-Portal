import sys
import pickle
import numpy as np
import streamlit as st
import sklearn
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Disease Prediction")
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

# loading the saved models
diabetes_model_1 = pickle.load(open(r"C:\Users\shamb\OneDrive\Documents\Major Project - EasyMed\Module 2\model1.sav", 'rb'))
diabetes_model_2 = pickle.load(open(r"C:\Users\shamb\OneDrive\Documents\Major Project - EasyMed\Module 2\model2.sav", 'rb'))
diabetes_model_3 = pickle.load(open(r"C:\Users\shamb\OneDrive\Documents\Major Project - EasyMed\Module 2\model3.sav", 'rb'))
diabetes_model_final = pickle.load(open(r"C:\Users\shamb\OneDrive\Documents\Major Project - EasyMed\Module 2\model_final.sav", 'rb'))

symptom_model_1 = pickle.load(open(r"C:\Users\shamb\OneDrive\Documents\Major Project - EasyMed\Module 1\model1.sav", 'rb'))

heart_model_1 = pickle.load(open(r"C:\Users\shamb\OneDrive\Documents\Major Project - EasyMed\Module 3\model1.sav", 'rb'))
heart_model_2 = pickle.load(open(r"C:\Users\shamb\OneDrive\Documents\Major Project - EasyMed\Module 3\model2.sav", 'rb'))
heart_model_3 = pickle.load(open(r"C:\Users\shamb\OneDrive\Documents\Major Project - EasyMed\Module 3\model3.sav", 'rb'))
heart_model_final = pickle.load(open(r"C:\Users\shamb\OneDrive\Documents\Major Project - EasyMed\Module 3\model_final.sav", 'rb'))


# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('EasyMed ',
                          
                          ['Diabetes Prediction',
                          'Symptoms Based Disease Prediction',
                           'Heart Disease Prediction'],
                          icons=['activity','person','heart'],
                          default_index=0)
    
    
# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction using ML')
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
    
    # code for Prediction
    diab_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Diabetes Test Result'):
        try:
            preds1 = diabetes_model_1.predict([[int(Pregnancies), int(Glucose), int(BloodPressure), int(SkinThickness), int(Insulin), float(BMI), float(DiabetesPedigreeFunction), int(Age)]])
            preds2 = diabetes_model_2.predict([[int(Pregnancies), int(Glucose), int(BloodPressure), int(SkinThickness), int(Insulin), float(BMI), float(DiabetesPedigreeFunction), int(Age)]])
            preds3 = diabetes_model_3.predict([[int(Pregnancies), int(Glucose), int(BloodPressure), int(SkinThickness), int(Insulin), float(BMI), float(DiabetesPedigreeFunction), int(Age)]])
            stack = np.column_stack((preds1,preds2,preds3))
            diab_prediction = diabetes_model_final.predict(stack)
            if (diab_prediction[0] == 1):
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'
        except:
            diab_diagnosis = 'Please check if input values are correct'
        
    st.success(diab_diagnosis)

if (selected == 'Symptoms Based Disease Prediction'):
    
    # page title
    st.title('Symptoms Based Disease Prediction using ML')

    # getting the input data from the user
    symtomns = ['itching', 'skin rash', 'nodal skin eruptions', 'continuous sneezing', 'shivering', 'chills', 'joint pain', 'stomach pain', 'acidity', 'ulcers on tongue', 'muscle wasting', 'vomiting', 'burning micturition', 'spotting urination', 'fatigue', 'weight gain', 'anxiety', 'cold hands and feets', 'mood swings', 'weight loss', 'restlessness', 'lethargy', 'patches in throat', 'irregular sugar level', 'cough', 'high fever', 'sunken eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish skin', 'dark urine', 'nausea', 'loss of appetite', 'pain behind the eyes', 'back pain', 'constipation', 'abdominal pain', 'diarrhoea', 'mild fever', 'yellow urine', 'yellowing of eyes', 'acute liver failure', 'fluid overload', 'swelling of stomach', 'swelled lymph nodes', 'malaise', 'blurred and distorted vision', 'phlegm', 'throat_irritation', 'redness of eyes', 'sinus pressure', 'runny nose', 'congestion', 'chest pain', 'weakness in limbs', 'fast heart rate', 'pain during bowel movements', 'pain in anal region', 'bloody stool', 'irritation in anus', 'neck pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen blood vessels', 'puffy face and eyes', 'enlarged thyroid', 'brittle_nails', 'swollen extremeties', 'excessive hunger', 'extra marital contacts', 'drying and tingling lips', 'slurred speech', 'knee pain', 'hip join tpain', 'muscle weakness', 'stiff neck', 'swelling joints', 'movement stiffness', 'spinning movements', 'loss of balance', 'unsteadiness', 'weakness of one body side', 'loss of smell', 'bladder discomfort', 'foul smell of urine', 'continuous feel of urine', 'passage of gases', 'internal itching', 'toxic look (typhos)', 'depression', 'irritability', 'muscle pain', 'altered sensorium', 'red spots over body', 'belly pain', 'abnormal menstruation', 'dischromic patches', 'watering from eyes', 'increased appetite', 'polyuria', 'family history', 'mucoid sputum', 'rusty sputum', 'lack of concentration', 'visual disturbances', 'receiving blood transfusion', 'receiving unsterile injections', 'coma', 'stomach bleeding', 'distention of abdomen', 'history of alcohol consumption', 'fluid overload', 'blood in sputum', 'prominent veins on calf', 'palpitations', 'painful walking', 'pus filled pimples', 'blackheads', 'scurring', 'skin peeling', 'silver like dusting', 'small dents in nails', 'inflammatory nails', 'blister', 'red sore around nose', 'yellow crust ooze']

    sym = st.multiselect(label='Select Symtomns', options=symtomns, default=None)
    
    # code for Prediction
    diagnosis = ''
    
    # creating a button for Prediction
    if st.button('Test Result'):
        inp = []
        for i in symtomns:
            if i in sym:
                inp.append(1)
            else:
                inp.append(0)
        
        pred = symptom_model_1.predict([inp])
        if (len(pred) == 1):
            diagnosis = 'The person has '+pred[0]
        if (len(pred)> 1):
            diagnosis = 'The person has '+str(pred)
        elif(len(pred)<1):
            diagnosis = 'The person don\'t have any disease'

    st.success(diagnosis)

# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
    # code for Prediction
    heart_diagnosis = ''
    # creating a button for Prediction
    if st.button('Heart Test Result'):
        try:
            preds1 = heart_model_1.predict([[int(age), int(sex), int(cp), int(trestbps), int(chol), int(fbs), int(restecg), int(thalach), int(exang), float(oldpeak), int(slope), int(ca), int(thal)]])
            preds2 = heart_model_2.predict([[int(age), int(sex), int(cp), int(trestbps), int(chol), int(fbs), int(restecg), int(thalach), int(exang), float(oldpeak), int(slope), int(ca), int(thal)]])
            preds3 = heart_model_3.predict([[int(age), int(sex), int(cp), int(trestbps), int(chol), int(fbs), int(restecg), int(thalach), int(exang), float(oldpeak), int(slope), int(ca), int(thal)]])

            stack = np.column_stack((preds1,preds2,preds3))
            heart_prediction = heart_model_final.predict(stack)
            if (heart_prediction[0] == 1):
                heart_diagnosis = 'The person has heart disease'
            else:
                heart_diagnosis = 'The person don\'t has heart disease'
        except:
            heart_diagnosis = 'Please check if input values are correct'
    st.success(heart_diagnosis)