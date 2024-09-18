import streamlit as st
import pandas as pd
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt

# Load the pre-trained image classification model (EfficientNetB0)
model = EfficientNetB0(weights="imagenet")

# Function to preprocess the image for the model
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# Function to load and clean the food dataset (local CSV file)
def load_food_dataset():
    dataset = pd.read_csv(r'C:\Users\Megha\OneDrive\Desktop\nutrients.csv')
    
    # Clean and convert columns to numeric values
    dataset['Calories'] = dataset['Calories'].replace({'cal': ''}, regex=True)
    dataset['Calories'] = pd.to_numeric(dataset['Calories'], errors='coerce').fillna(0)
    dataset['Grams'] = pd.to_numeric(dataset['Grams'], errors='coerce').fillna(0)
    dataset['Protein'] = pd.to_numeric(dataset['Protein'], errors='coerce').fillna(0)
    dataset['Fat'] = pd.to_numeric(dataset['Fat'], errors='coerce').fillna(0)
    dataset['Sat.Fat'] = pd.to_numeric(dataset['Sat.Fat'], errors='coerce').fillna(0)
    dataset['Fiber'] = pd.to_numeric(dataset['Fiber'], errors='coerce').fillna(0)
    dataset['Carbs'] = pd.to_numeric(dataset['Carbs'], errors='coerce').fillna(0)
    
    return dataset

# Function to classify food using the pre-trained model
def classify_food(img):
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)
    
    # Decode the predictions to get human-readable labels
    decoded_predictions = decode_predictions(predictions, top=5)[0]
    
    # Get the top prediction
    food_label = decoded_predictions[0][1]
    confidence = decoded_predictions[0][2]
    return food_label, confidence

# Function to get nutritional information based on identified food item
def get_nutritional_info(food_item, dataset):
    row = dataset[dataset['Food'].str.contains(food_item, case=False, na=False)]
    
    if not row.empty:
        return {
            'measure': row['Measure'].values[0],
            'grams': row['Grams'].values[0],
            'calories': row['Calories'].values[0],
            'protein': row['Protein'].values[0],
            'fat': row['Fat'].values[0],
            'sat_fat': row['Sat.Fat'].values[0],
            'fiber': row['Fiber'].values[0],
            'carbs': row['Carbs'].values[0]
        }
    else:
        return {
            'measure': 'N/A',
            'grams': 'N/A',
            'calories': 'N/A',
            'protein': 'N/A',
            'fat': 'N/A',
            'sat_fat': 'N/A',
            'fiber': 'N/A',
            'carbs': 'N/A'
        }

# Function to plot a pie chart for all nutrients
def plot_nutrition_pie_chart(nutrition_info):
    labels = ['Calories', 'Protein', 'Fat', 'Sat.Fat', 'Fiber', 'Carbs']
    sizes = [
        nutrition_info.get('calories', 0),
        nutrition_info.get('protein', 0),
        nutrition_info.get('fat', 0),
        nutrition_info.get('sat_fat', 0),
        nutrition_info.get('fiber', 0),
        nutrition_info.get('carbs', 0)
    ]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6']
    
    # Replace NaN and 'N/A' values with 0 in sizes
    sizes = [float(size) if not pd.isna(size) and size != 'N/A' else 0 for size in sizes]
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2.5)))])
    fig.update_layout(title_text='Nutritional Breakdown')
    return fig

# Function to plot top 20 calorie-rich foods
def plot_top_20_calorie_rich_foods(dataset):
    # Sort and get top 20 calorie-rich foods
    dataset['Calories'] = dataset['Calories'].fillna(0)  # Fill NaNs with 0 before sorting
    cals = dataset.sort_values(by='Calories', ascending=False)
    top_20_cals = cals.head(20)
    
    # Create a bar chart using Plotly
    fig = px.bar(top_20_cals, x='Food', y='Calories', color='Calories', title='Top 20 Calorie-Rich Foods')
    fig.update_layout(xaxis_title='Food', yaxis_title='Calories')
    return fig

# Function to plot top 20 fat content and calories
def plot_top_20_fat_content(dataset):
    # Sort and get top 20 foods by fat content
    dataset['Fat'] = dataset['Fat'].fillna(0)  # Fill NaNs with 0 before sorting
    fats = dataset.sort_values(by='Fat', ascending=False)
    top_20_fat = fats.head(20)
    
    # Create a bar chart using Plotly
    fig = px.bar(top_20_fat, x='Food', y='Calories', color='Calories', title='Fat Content and Calories')
    fig.update_layout(xaxis_title='Food', yaxis_title='Calories')
    return fig

# Function to plot 3D scatter plot of carbohydrate-rich foods
def plot_3d_scatter_plot(dataset):
    trace1 = go.Scatter3d(
        x=dataset['Category'].values,
        y=dataset['Food'].values,
        z=dataset['Carbs'].values,
        text=dataset['Food'].values,
        mode='markers',
        marker=dict(
            sizemode='diameter',
            sizeref=750,
            color=dataset['Carbs'].values,
            colorscale='Portland',
            colorbar=dict(title='Carbohydrates'),
            line=dict(color='rgb(255, 255, 255)')
        )
    )
    data = [trace1]
    layout = dict(height=800, width=800, title='3D Scatter Plot of Carbohydrate Rich Food')
    fig = dict(data=data, layout=layout)
    return fig

# Function to plot boxen plot for calorie content by category
def plot_calorie_boxen_plot(dataset):
    plt.figure(figsize=(22,10))
    ax = sns.boxenplot(x="Category", y='Calories', data=dataset, color='#eeeeee', palette="tab10")

    # Add transparency to colors
    for patch in ax.artists:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, .9))
    
    plt.title("Total Calorie Content \n", loc="center", size=32, color='#be0c0c', alpha=0.6)
    plt.xlabel('Category', color='#34495E', fontsize=20) 
    plt.ylabel('Total Calories', color='#34495E', fontsize=20)
    plt.xticks(size=16, color='#008abc', rotation=90, wrap=True)  
    plt.yticks(size=15, color='#006600')
    plt.tight_layout()
    return plt

# Streamlit app initialization
st.set_page_config(page_title="AI Food Calorie Estimator")
st.header("Food Calorie Estimator Using AI")

# Image uploader for food image
uploaded_file = st.file_uploader("Choose an image of food...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    submit = st.button("Identify Food and Estimate Nutritional Information")
    
    if submit:
        # Load and clean the dataset
        food_dataset = load_food_dataset()
        
        # Classify food from the image using the pre-trained model
        identified_food, confidence = classify_food(img)
        
        # Get nutritional information
        results = get_nutritional_info(identified_food, food_dataset)
        
        # Display the results
        st.subheader("Nutritional Information")
        st.markdown(f"### Food Item: **{identified_food.capitalize()}**")
        st.markdown(f"**Calories:** <span style='font-size:24px; color:red;'>{results['calories']} kcal</span>", unsafe_allow_html=True)
        st.write(f"**Measure:** {results['measure']}")
        st.write(f"**Grams:** {results['grams']} g")
        st.write(f"**Protein:** {results['protein']} g")
        st.write(f"**Fat:** {results['fat']} g")
        st.write(f"**Saturated Fat:** {results['sat_fat']} g")
        st.write(f"**Fiber:** {results['fiber']} g")
        st.write(f"**Carbohydrates:** {results['carbs']} g")
        st.write(f"**Confidence:** {confidence:.2%}")
        
        # Plot and display pie chart for all nutrients
        st.subheader("Nutritional Breakdown")
        nutrition_chart = plot_nutrition_pie_chart(results)
        st.plotly_chart(nutrition_chart)
        
        # Plot and display top 20 calorie-rich foods bar chart
        st.subheader("Top 20 Calorie-Rich Foods")
        top_20_chart = plot_top_20_calorie_rich_foods(food_dataset)
        st.plotly_chart(top_20_chart)
        
        # Plot and display top 20 fat content bar chart
        st.subheader("Top 20 Fat Content and Calories")
        top_20_fat_chart = plot_top_20_fat_content(food_dataset)
        st.plotly_chart(top_20_fat_chart)
        
        # Plot and display 3D scatter plot for carbohydrate-rich foods
        st.subheader("3D Scatter Plot of Carbohydrate-Rich Foods")
        scatter_3d_fig = plot_3d_scatter_plot(food_dataset)
        st.plotly_chart(scatter_3d_fig)
        
        # Plot and display boxen plot for calorie content by category
        st.subheader("Calorie Content by Category")
        calorie_boxen_fig = plot_calorie_boxen_plot(food_dataset)
        st.pyplot(calorie_boxen_fig)
        
        # Add a feedback section
        st.subheader("Feedback")
        feedback = st.text_area("Was this information helpful? Let us know!")
        
        if st.button("Submit Feedback"):
            st.write("Thank you for your feedback!")
