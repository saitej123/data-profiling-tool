#!/usr/bin/env python
# coding: utf-8

# Author: Sai
import os
import numpy as np
import pandas as pd
#from pycaret.classification import *
#from pycaret.regression import *
from joblib import load
import streamlit as st
import _pickle as pickle
from pprint import pformat
from PIL import Image
import markdown
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, HTML
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

#custom fucntions
def show_columns(x):
    return x.columns

def pycaret_automl(x,y):
    setting=setup(x, target =y )
    # compare all baseline models and select top 5
    #top5 = compare_models(n_select = 5)
    # tune top 5 base models
    #tuned_top5 = [tune_model(i) for i in top5]
    # ensemble top 5 tuned models
    #bagged_top5 = [ensemble_model(i) for i in tuned_top5]
    # blend top 5 base models
    #blender = blend_models(estimator_list = top5)
    # select best model
    #best = automl(optimize = 'Recall')
    return st.write(setting)

def main():
        image = Image.open('sai_app_header.png')
        im = Image.open('data_profile.png')
        st.image(im,use_column_width=None)
        st.subheader("Data Profiling Tool")
        st.sidebar.image(image,use_column_width=None)
        tasks = ["EDA Analysis","ML Model Building with Pycaret"]
        choice = st.sidebar.selectbox("Select Task To do",tasks)
        st.set_option('deprecation.showfileUploaderEncoding', False)

        data = st.file_uploader("Upload a Dataset (CSV or TXT)", type=["csv", "txt"])
        if data is not None:
            df = pd.read_csv(data)
            st.success("Your Data Frame Loaded successfully")
            all_columns = show_columns(df)
            if choice == 'EDA Analysis':
                if st.button("Generate Pandas Profiling Report"):
                        pr = ProfileReport(df, explorative=True)
                        st.subheader("Pandas Profiling : Quick Exploratory data analysis")
                        st.text('We Get :')
                        st.text('DataFrame overview')
                        st.text('Each attribute on which DataFrame is defined')
                        st.text('Correlations between attributes')
                        st.text('A sample of DataFrame')
                        st_profile_report(pr)


            if choice == 'ML Model Building with Pycaret':
                target_selection= st.selectbox("Select Target Variable Column",all_columns)

                if st.button("Run Pycaret AutoML"):
                        pycaret_automl(df,target_selection)









if __name__ == '__main__':
    main()
