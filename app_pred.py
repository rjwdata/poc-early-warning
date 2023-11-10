import pickle
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import streamlit as st
from PIL import Image
import yaml
from src.pipeline.predict_pipeline import PredictPipeline
from src.utils import load_object

transformer_path = os.path.join("artifacts", "proprocessor.pkl")
model_path = os.path.join("artifacts", "model.pkl")
# Load  model a

model = joblib.load(model_path)
preprocessor = joblib.load(transformer_path)

def visualize_confidence_level(prediction_proba):
    """
    this function uses matplotlib to create inference bar chart rendered with streamlit in real-time
    return type : matplotlib bar chart
    """
    data = (prediction_proba[0]*100).round(2)
    grad_percentage = pd.DataFrame(data = data,columns = ['Percentage'],index = ['Low','Ave','High'])
    ax = grad_percentage.plot(kind='barh', figsize=(7, 4), color='#722f37', zorder=10, width=0.5)
    ax.legend().set_visible(False)
    ax.set_xlim(xmin=0, xmax=100)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    ax.set_xlabel(" Percentage(%) Confidence Level", labelpad=2, weight='bold', size=12)
    ax.set_ylabel("Wine Quality", labelpad=10, weight='bold', size=12)
    ax.set_title('Prediction Confidence Level ', fontdict=None, loc='center', pad=None, weight='bold')

    st.pyplot()
    return

st.title('POC Early Warning System')

st.markdown("""
Currently, we employ a one-size-fits-all system for identifying at-risk students, using indicators such as attendance below 90%, the number of D's and F's in grades, and the frequency of suspensions in discipline records. However, this approach lacks specificity regarding what students are at risk of, be it graduation, college admission, or success on college placement tests.

The primary purpose of project is to leverage client data to create individualized indicators of students at-risk of not graduating. This versatile framework which can serve as a template for predicting various outcomes whether they are continuous, binary, or multiclass.

## Instructions
1.  Select the features from the sidebar
2.  Model will provide a prediction at the bottom of the page


## Dataset Source :


You can also :
* Check the **GitHub Project Repository**
            [![](https://img.shields.io/badge/POC%20Early%20Warning-GitHub-100000?logo=github&logoColor=white)](https://github.com/rjwdata/poc-early-warning)
""")
df = pd.read_csv(os.path.join('data','raw','train','train.csv'))

st.header('Student Data Overview')
st.write('Data Dimension: ' + str(df.shape[0]) + ' rows and ' + str(df.shape[1]) + ' columns.')
st.dataframe(df)

#read in wine image and render with streamlit
#image = Image.open('wine_image.png')
#st.image(image, caption='wine company',use_column_width=True)

st.sidebar.header('User Input Parameters') #user input parameter collection with streamlit side bar


def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox
    return type : pandas dataframe

    """
    male = st.sidebar.selectbox("Select Male",("yes", "no"))
    race_ethnicity = st.sidebar.selectbox("Select Race/Ethnicity",('African-American', 'Asian/Pacific Islander', 'Hispanic', 'Multiple/Native American', 'Other'))
    iep = st.sidebar.selectbox("Select IEP Services",('yes', 'no'))
    frpl = st.sidebar.selectbox("Select FRPL Services",('yes', 'no'))
    ell = st.sidebar.selectbox("Select ELL Services",('yes', 'no'))
    ap_ever_take_class = st.sidebar.selectbox("Select AP Ever",('yes', 'no'))
    ever_alternative = st.sidebar.selectbox("Select Alternate Ever",('yes', 'no'))
    gpa = st.sidebar.slider('Select GPA', 0.0, 4.0, 2.80)
    math_ss = st.sidebar.slider('Select Math SS', 0, 200, 50)
    read_ss = st.sidebar.slider('Select Read SS', 0, 200, 50)
    pct_days_absent = st.sidebar.slider('Select Percentage of Days Missed', 0.0, 100.0, 8.5)
    scale_score_11_comp = st.sidebar.slider('Select ACT Composite Score', 0.0, 36.0, 19.0)
    scale_score_11_eng = st.sidebar.slider('Select ACT English Score', 0.0, 36.0, 19.0)
    scale_score_11_math = st.sidebar.slider('Select ACT Math Score', 0.0, 36.0, 19.0)
    scale_score_11_read = st.sidebar.slider('Select ACT Reading Score', 0.0, 36.0, 20.0)

    features = {
            'male': male,
            'race_ethnicity':race_ethnicity,
            'iep':iep,
            'frpl':frpl,
            'ell':ell,
            'ap_ever_take_class':ap_ever_take_class,
            'ever_alternative':ever_alternative,
            'gpa':gpa,
            'pct_days_absent':pct_days_absent,
            'math_ss':math_ss,
            'read_ss':read_ss,
            'scale_score_11_comp':scale_score_11_comp,
            'scale_score_11_eng':scale_score_11_eng,
            'scale_score_11_math':scale_score_11_math,
            'scale_score_11_read':scale_score_11_read
            }
    data = pd.DataFrame(features,index=[0])

    return data


user_input_df = get_user_input()

predict_pipeline = PredictPipeline()
prediction = predict_pipeline.predict(user_input_df)
# params_path = "params.yaml"

# def read_params(config_path = params_path):
    # with open(config_path) as yaml_file:
        # config = yaml.safe_load(yaml_file)
    # return config

# def predict(data):
    # config = read_params(params_path)
    # model_dir_path = config["webapp_model_dir"]
    # model = joblib.load(model_dir_path)
    # prediction = model.predict(data)
    # return prediction

st.subheader('User Input parameters')
st.write(user_input_df)

if prediction[0] == 1:
    grad = 'Diploma'
else:
    grad = 'No Diploma'

hs_grad = '{:.2f}%'.format(prediction[1][0][1]*100)
no_hs_grad = '{:.2f}%'.format(prediction[1][0][0]*100)

col1, col2, col3 = st.columns(3)
col1.metric("Diploma Prediction", grad)
col2.metric('Probability of HS Diploma', hs_grad)
col3.metric('Prbability of no HS Diploma', no_hs_grad)

#prediction_proba = model.predict_prob(user_input_df)

#visualize_confidence_level(prediction_proba)
