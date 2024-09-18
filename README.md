The AI Food Calorie Estimator is a Streamlit-based web application that estimates the nutritional content of food items using image recognition and a pre-trained deep learning model (EfficientNetB0). The app also provides detailed nutritional information based on a local dataset of food items and visualizes the top calorie-rich foods, carbohydrate-rich foods, and category-wise calorie distribution.


## Features

- **Image-Based Food Identification**:  
  Upload a food image, and the app will classify the food item using the **EfficientNetB0** model pre-trained on **ImageNet**.

- **Nutritional Breakdown**:  
  Provides detailed nutritional information including **calories**, **protein**, **fat**, **carbs**, and other nutrients for the identified food item, based on a local dataset.

- **Visualizations**:
  - **Pie Chart**: Displays the nutritional breakdown of the identified food item.
  - **Bar Chart**: Shows the top 20 calorie-rich foods from the dataset.
  - **3D Scatter Plot**: Visualizes carbohydrate-rich foods by category.
  - **Boxen Plot**: Represents the calorie content distribution by food category.

- **User Feedback**:  
  Users can submit feedback on the identified food and its nutritional content.
