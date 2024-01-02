import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
scaler = StandardScaler()
import joblib
import os
import time
import threading

df = pd.read_csv('dataset/Kaagle_Upload.csv',low_memory=False)

excel_file_path = 'dataset/Road-Accident-Safety-Data-Guide (1).xlsx'

# Read the Excel file with specific sheet names
xls = pd.ExcelFile(excel_file_path)

# Get the list of sheet names from the Excel file
sheet_names = xls.sheet_names

st.title("Road Accident Analysis and Severity Prediction")
page = st.sidebar.selectbox("Select Page", ["Analysis", "SVC Model","Desision Tree Model","Logistic Regression"])
    

# Display the first few rows of the dataset
if page == "Analysis":
  st.markdown("<p style='color:red;'>Don't Use Mobile While Driving Can Cause Accidents</p>", unsafe_allow_html=True)
  st.text("Please drive safely!")
  images = ["images/mobile.jpg", "images/accident.jpg"]

# Set the current image index
  current_image_index = 0

# Create a st.empty() container
  image_container = st.empty()

# Display the current image in the st.empty() container
  image_container.image(images[current_image_index])
  
# Start a timer to switch the image every 5 seconds
  def while_loop_function():
      current_image_index = 0
      while True:
         
         st.session_state.current_image_index = current_image_index
    # Switch the image in the st.empty() container
         image_container.image(images[current_image_index])

    # Increment the current image index
         current_image_index += 1

    # If the current image index is greater than the length of the images list, reset it to 0
         current_image_index %= len(images)

    # Wait for 1 second
         time.sleep(5)
          
# Create a thread to run the while loop in the background
  
  

# Start the thread
  st.write("### Preview of the Dataset")
  st.write(df.head(20))
  
# Data Preprocessing
# ... (your preprocessing code here)
  df2 = df[['special_conditions_at_site','number_of_casualties','road_surface_conditions','light_conditions','weather_conditions','age_of_vehicle','sex_of_driver','age_of_driver',
'junction_location', 'junction_detail','junction_control','casualty_severity',
  'accident_severity','day_of_week']]
  df2.replace(-1, np.nan, inplace=True) 
  df2=df2.dropna()
  df2.shape
# Correlation Heatmap
  st.header('Correlation Heatmap')
  st.write('Heatmap showing correlations between selected variables')
  st.write("(Correlation heatmap is computed based on selected features)")

  corrmat = df2.corr()
  k = 6 
  cols = corrmat.nlargest(k, 'accident_severity')['accident_severity'].index
  cm = np.corrcoef(df2[cols].values.T)


# Display the heatmap
  fig, ax = plt.subplots(figsize=(10, 8))
  sns.set(font_scale=1.25)
  hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values, ax=ax)
  st.pyplot(fig)
  df=df.dropna()
  df=df.dropna()

  st.write("Before Preprocessing")
  
  st.write("### Histogram of 'Age of driver'")
  plt.figure(figsize=(6, 4))
  plt.hist(df2['age_of_driver'], bins=30, color='skyblue', alpha=0.7)
  plt.xlabel('Age of Driver')
  plt.ylabel('Frequency')
  hist_fig = plt.gcf()  # Get the current figure
  st.pyplot(hist_fig)
# Probability Plot
  st.write("### Probability Plot of 'age_of_driver'")
  fig = plt.figure()
  res = stats.probplot(df2['age_of_driver'], plot=plt)
  st.pyplot(fig)

  
  df2['age_of_driver'] = np.log1p(df2['age_of_driver']) 
  df2['age_of_vehicle'] = np.log1p(df2['age_of_vehicle'])
  st.write("After Preprocessing")
 
  

# Probability Plot
  st.write("### Probability Plot of 'age_of_driver'")
  fig = plt.figure()
  res = stats.probplot(df2['age_of_driver'], plot=plt)
  st.pyplot(fig)  
  st.write("### Map of Accident")
  st.map(df,
    latitude=df['latitude'],
    longitude=df['longitude'])
  
  st.write("### Histogram of 'Age of driver'")
  plt.figure(figsize=(6, 4))
  plt.hist(df2['age_of_driver'], bins=30, color='skyblue', alpha=0.7)
  plt.xlabel('Age of Driver')
  plt.ylabel('Frequency')
  hist_fig = plt.gcf()  # Get the current figure
  st.pyplot(hist_fig)

  st.write("### Bargraph of 'Number of Accidents and the Day on which it occured'")
  day_labels = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

  plt.figure(figsize=(8, 6))
  sns.countplot(x='day_of_week', data=df2)
  plt.xlabel('Day of Week')
  plt.ylabel('Count')
  plt.xticks(ticks=range(7), labels=day_labels)  # Set custom day labels
  bar_fig = plt.gcf()  # Get the current figure
  st.pyplot(bar_fig)
  while_loop_thread = threading.Thread(target=while_loop_function())
  


elif page == "SVC Model":
    df2 = df[['special_conditions_at_site','number_of_casualties','road_surface_conditions','light_conditions','weather_conditions','age_of_vehicle','sex_of_driver','age_of_driver',
'junction_location', 'junction_detail','junction_control','casualty_severity',
     'accident_severity','day_of_week']]
    df2.replace(-1, np.nan, inplace=True) 
    df2=df2.dropna()
    st.markdown("Standard Vector ")
# the svc
# Machine Learning Model

    scaler = StandardScaler()
    svc = SVC()
    df2=df2[:15000]
    data = {}
# Select features for prediction    
    selected_features = st.multiselect('Select Features for Prediction1', df2.columns.drop('accident_severity'))
    X = df2[selected_features]
    Y = df2['accident_severity']

# Train-test split ratio
    test_size = st.slider('Test Data Ratio', 0.1, 0.5, 0.2, 0.05)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=99)
# Model Training
# ... (previous code remains unchanged) ...

# Model Training
# Model Training
    if st.button('Train Model'):
    # Standardize the features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

    # Train the model
        svc.fit(X_train_scaled, Y_train)
    # Make predictions
        Y_pred = svc.predict(X_test_scaled)

    # Calculate accuracy
        accuracy = accuracy_score(Y_test, Y_pred)
        st.write('**Accuracy:**', accuracy)

    # Save the trained model and fitted scaler to files
        joblib.dump(svc, 'trained_model.pkl')
        joblib.dump(scaler, 'fitted_scaler.pkl')
        st.write('Model Trained and Saved Successfully!')
        st.write('Comparasion between Original and Predicted')
        comparison_tuples = list(zip(Y_test, Y_pred))

# Display a specific number of samples (e.g., 10 samples) from the tuples in a table
        num_samples_to_display = 10
        st.table(comparison_tuples[:num_samples_to_display])

# Make Predictions
    st.header('Make Predictions')
    st.write('### Enter Feature Values for Prediction')
    input_values = {}
    sheet_names = pd.ExcelFile(xls).sheet_names
    for feature in selected_features:
        for sheet_name in sheet_names:
            if(sheet_name==feature):
    # Read the data from the current sheet
                sheet_data = pd.read_excel(xls, sheet_name)
    # Display the table
                st.write(sheet_data)
        input_values[feature] = st.number_input(f'Enter {feature}', min_value=0)        
# Iterate over the sheet names and display the table for the current sheet
        

# Loop through each sheet and read data into a DataFrame
    

# Make prediction
    if st.button('Predict'):
    # Check if the model and scaler files exist
     if 'trained_model.pkl' in os.listdir() and 'fitted_scaler.pkl' in os.listdir():

        # Load the trained model and fitted scaler
        saved_model = joblib.load('trained_model.pkl')
        fitted_scaler = joblib.load('fitted_scaler.pkl')

        # Prepare input data for prediction
        input_data = pd.DataFrame([input_values])

        # Standardize input data using the loaded fitted scaler
        input_data_scaled = fitted_scaler.transform(input_data)

        # Make prediction using the loaded model
        prediction = saved_model.predict(input_data_scaled)
        severity= prediction[0]
        if (severity==1):
            st.write('**Predicted Accident Severity:** is Fatal')
        elif (severity==2):
            st.write('**Predicted Accident Severity:** is Serious')
        else:
            st.write('**Predicted Accident Severity:** is Slight')
     else:
        st.write('Please train the model first by clicking the "Train Model" button.')

elif page =="Desision Tree Model":

    # Data Preprocessing
    df2 = df[['special_conditions_at_site','number_of_casualties','road_surface_conditions','light_conditions','weather_conditions','age_of_vehicle','sex_of_driver','age_of_driver',
'junction_location', 'junction_detail','junction_control','casualty_severity',
 'accident_severity','day_of_week']]
    df2.replace(-1, np.nan, inplace=True) 
    df2 = df2.dropna()

# Streamlit App

# Machine Learning Model Setup
    scaler = StandardScaler()
    decision_tree = DecisionTreeClassifier()

# Select Features for Prediction
    selected_features_decision_tree = st.multiselect('Select Features for Prediction (Decision Tree)', df2.columns.drop('accident_severity'))
    X = df2[selected_features_decision_tree]
    Y = df2['accident_severity']

# Train-test split ratio
    test_size = st.slider('Test Data Ratio', 0.1, 0.5, 0.2, 0.05)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=99)

# Model Training (Decision Tree)
    if st.button('Train Decision Tree Model'):
         
         X_train_scaled_dt = scaler.fit_transform(X_train)
         X_test_scaled_dt = scaler.transform(X_test)

         decision_tree.fit(X_train_scaled_dt, Y_train)
         Y_pred_dt = decision_tree.predict(X_test_scaled_dt)
         accuracy_dt = accuracy_score(Y_test, Y_pred_dt)
         st.write('**Decision Tree Model Accuracy:**', accuracy_dt)
         joblib.dump(decision_tree, 'decision_tree_trained_model.pkl')
         joblib.dump(scaler, 'dt_fitted_scaler.pkl')
         st.write('Decision Tree Model Trained and Saved Successfully!')
         st.write('Comparasion between Original and Predicted')
         comparison_tuples = list(zip(Y_test, Y_pred_dt))

# Display a specific number of samples (e.g., 10 samples) from the tuples in a table
         num_samples_to_display = 10
         st.table(comparison_tuples[:num_samples_to_display])

# Make Predictions
    st.header('Make Predictions')
    st.write('### Enter Feature Values for Prediction')
    input_values = {}
    sheet_names = pd.ExcelFile(xls).sheet_names
    for feature in selected_features_decision_tree:
        for sheet_name in sheet_names:
            if(sheet_name==feature):
    # Read the data from the current sheet
                sheet_data = pd.read_excel(xls, sheet_name)
    # Display the table
                st.write(sheet_data)
        input_values[feature] = st.number_input(f'Enter {feature}', min_value=0)    

# Make Decision Tree prediction
    if st.button('Predict (Decision Tree)'):
     if 'decision_tree_trained_model.pkl' in os.listdir() and 'dt_fitted_scaler.pkl' in os.listdir():
        saved_model_dt = joblib.load('decision_tree_trained_model.pkl')
        fitted_scaler_dt = joblib.load('dt_fitted_scaler.pkl')

        input_data_dt = pd.DataFrame([input_values])
        input_data_scaled_dt = fitted_scaler_dt.transform(input_data_dt)
        prediction_dt = saved_model_dt.predict(input_data_scaled_dt)[0]
        
        if prediction_dt == 1:
            st.write('**Predicted Accident Severity (Decision Tree):** Fatal')
        elif prediction_dt == 2:
            st.write('**Predicted Accident Severity (Decision Tree):** Serious')
        else:
            st.write('**Predicted Accident Severity (Decision Tree):** Slight')
     else:
        st.write('Please train the Decision Tree model first by clicking the "Train Decision Tree Model" button.')

elif page =="Logistic Regression":
   df2 = df[['special_conditions_at_site','number_of_casualties','road_surface_conditions','light_conditions','weather_conditions','age_of_vehicle','sex_of_driver','age_of_driver',
'junction_location', 'junction_detail','junction_control','casualty_severity',
 'accident_severity','day_of_week']]
   df2.replace(-1, np.nan, inplace=True) 
   df2 = df2.dropna()

# Machine Learning Model Setup
   scaler = StandardScaler()
   logreg = LogisticRegression()

# Select Features for Prediction
   selected_features_logreg = st.multiselect('Select Features for Prediction (Logistic Regression)', df2.columns.drop('accident_severity'))
   X = df2[selected_features_logreg]
   Y = df2['accident_severity']

# Train-test split ratio
   test_size = st.slider('Test Data Ratio', 0.1, 0.5, 0.2, 0.05)
   X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=99)

# Model Training (Logistic Regression)
   if st.button('Train Logistic Regression Model'):
       X_train_scaled_logreg = scaler.fit_transform(X_train)
       X_test_scaled_logreg = scaler.transform(X_test)

       logreg.fit(X_train_scaled_logreg, Y_train)
       Y_pred_logreg = logreg.predict(X_test_scaled_logreg)  
       accuracy_logreg = accuracy_score(Y_test, Y_pred_logreg)
       st.write('**Logistic Regression Model Accuracy:**', accuracy_logreg)
       joblib.dump(logreg, 'logreg_trained_model.pkl')
       joblib.dump(scaler, 'logreg_fitted_scaler.pkl')
       st.write('Logistic Regression Model Trained and Saved Successfully!')
       st.write('Comparasion between Original and Predicted')
       comparison_tuples = list(zip(Y_test, Y_pred_logreg))

# Display a specific number of samples (e.g., 10 samples) from the tuples in a table
       num_samples_to_display = 10
       st.table(comparison_tuples[:num_samples_to_display])

# Make Predictions
   st.header('Make Predictions')
   st.write('### Enter Feature Values for Prediction')
   input_values = {}
   sheet_names = pd.ExcelFile(xls).sheet_names
   for feature in selected_features_logreg:
       for sheet_name in sheet_names:
           if(sheet_name==feature):
    # Read the data from the current sheet
               sheet_data = pd.read_excel(xls, sheet_name)
    # Display the table
               st.write(sheet_data)
       input_values[feature] = st.number_input(f'Enter {feature}', min_value=0)    

# Make Logistic Regression prediction
   if st.button('Predict (Logistic Regression)'):
       if 'logreg_trained_model.pkl' in os.listdir() and 'logreg_fitted_scaler.pkl' in os.listdir():
           saved_model_logreg = joblib.load('logreg_trained_model.pkl')
           fitted_scaler_logreg = joblib.load('logreg_fitted_scaler.pkl')

           input_data_logreg = pd.DataFrame([input_values])
           input_data_scaled_logreg = fitted_scaler_logreg.transform(input_data_logreg)
           prediction_logreg = saved_model_logreg.predict(input_data_scaled_logreg)[0]
        
       if prediction_logreg == 1:
           st.write('**Predicted Accident Severity (Logistic Regression):** Fatal')
       elif prediction_logreg == 2:
           st.write('**Predicted Accident Severity (Logistic Regression):** Serious')
       else:
           st.write('**Predicted Accident Severity (Logistic Regression):** Slight')
   else:
       st.write('Please train the Logistic Regression model first by clicking the "Train Logistic Regression Model" button.')
         