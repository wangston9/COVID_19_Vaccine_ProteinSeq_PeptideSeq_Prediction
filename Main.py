import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf

# Image
st.image("covid-vax-bottle-700x467.jpeg", output_format='auto')

st.title("COVID-19 Vaccine Peptide Sequence Prediction")
st.write("This app predicts potential COVID-19 epitope peptide sequence using Deep Neural Network.")
st.markdown("**Application:** Vaccine Development")
st.markdown("**Data Source:** [COVID-19/SARS B-cell Epitope Prediction](https://www.kaggle.com/datasets/futurecorporation/epitope-prediction)")
st.markdown("**Data License:** Attribution 4.0 International (CC BY 4.0)")
st.markdown("Information on whether or not an amino acid peptide exhibited antibody-inducing activity (marked by an activity label) could be obtained from IEDB, which was used in many previous studies. Accordingly, this information was used as the label data. We also obtained the epitope candidate amino acid sequences (peptides) and the activity label data from the B-cell epitope data provided in IEDB. The presented antibody proteins were restricted to IgG that constituted the most recorded type in IEDB.")
st.markdown("There are Three Datasets Available: B_Cell, SARS, COVID. B_Cell and SARS are labeled and combined to be used as Training Dataset. COVID dataset is used as testing dataset for prediction.")

# Image
st.image("BCell_Activation.jpeg", output_format='auto')

st.markdown("<h2 style='text-align: center;'>Data Properties:</h2>", unsafe_allow_html=True)
st.image("Data_Properties.png", output_format='auto')

st.markdown("<h2 style='text-align: center;'>Overview of Process:</h2>", unsafe_allow_html=True)
st.image("Process_Overview.png", output_format='auto')


# Load the cleaned data from the CSV file
data = pd.read_csv('COVID_TrainingData.csv', index_col=0, encoding= 'unicode_escape')

# Example: Display the first few rows of the loaded data
st.markdown("<h2 style='text-align: center;'>Sample of Training Dataset:</h2>", unsafe_allow_html=True)
st.write(data.head())
st.write("**Shape of Training Dataset:**", data.shape)

if st.checkbox("Show Statistics_Training Data"):
    # Center-align the statistics table
    st.markdown("<div style='text-align: center;'>Statistics for Training Data:</div>", unsafe_allow_html=True)
    st.table(data.describe())

# Load the cleaned data from the CSV file
data1 = pd.read_csv('COVID_TestData.csv', index_col=0, encoding= 'unicode_escape')

# Example: Display the first few rows of the loaded data
st.markdown("<h2 style='text-align: center;'>Sample of Testing Dataset:</h2>", unsafe_allow_html=True)
st.write(data1.head())
st.write("**Shape of Testing Dataset:**", data1.shape)
if st.checkbox("Show Statistics_Testing Data"):
    # Center-align the statistics table
    st.markdown("<div style='text-align: center;'>Statistics for Testing Data:</div>", unsafe_allow_html=True)
    st.table(data1.describe())

# Load your DNN model
model = tf.keras.models.load_model('covid_DNN.h5')

# Load the COVID input data
covid_input = pd.read_excel('input_covid.xlsx', sheet_name='Sheet 1') 

# Assuming you want to select the first 9 columns
covid_data = data1.iloc[:, :9].values

# Next, preprocess the data if necessary
def preprocess(covid_data):
    return np.array(covid_data, dtype=np.float32)

# Make predictions with Streamlit
st.title('COVID-19 Prediction App')

# Input field for the number of rows to display
num_rows = st.number_input('Number of Rows to Display', min_value=1, max_value=len(data1), value=5)

if st.button('Show Predictions'):
    covid_data = preprocess(covid_data)
    predictions = model.predict(covid_data)
    top_indices = np.argsort(predictions[:, 1])[-num_rows:][::-1]  # Get indices of highest predictions for Class1
    st.write(f'Top {num_rows} Predictions for Class1(Positive B-Cell Immune Reponse):')
    for idx in top_indices:
        # Get the corresponding row from COVID_input
        row_data = covid_input.iloc[idx]

        # Extract relevant features from COVID_input
        protein_seq = row_data['protein_seq']
        peptide_seq = row_data['peptide_seq']

        # Display features and Class1 probability
        st.write(f'Row {idx + 1}: Protein Seq: {protein_seq}, Peptide Seq: {peptide_seq}, Probability of Class1: {predictions[idx][1]}')
