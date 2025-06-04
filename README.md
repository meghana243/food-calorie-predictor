🥗 AI Food Calorie Estimator
An AI-powered web application built with Streamlit, TensorFlow, and Plotly to identify food items from images and estimate their nutritional values. It also visualizes nutritional insights from a curated food dataset.

📸 Features
🍕 Image-Based Food Identification using EfficientNetB0 (ImageNet pre-trained).

🔬 Nutritional Estimation from a custom CSV dataset (nutrients.csv).

📊 Dynamic Visualizations:

Nutritional pie chart for identified food

Top 20 calorie-rich foods bar chart

Top 20 fat content vs calories chart

3D scatter plot of carbohydrate-rich foods

Boxen plot of calories by food category

🗣️ User Feedback Section to collect feedback

🧠 Model Used
EfficientNetB0 (Keras Applications)

Pre-trained on ImageNet

Used for top-1 prediction from uploaded food images

📁 Dataset
Custom CSV file: nutrients.csv

Contains:

Food name

Category

Nutritional values: Calories, Protein, Fat, Saturated Fat, Fiber, Carbs

Grams and measure information

📈 Visualizations
Chart	Description
🍩 Pie Chart	Nutritional breakdown of the identified food
📉 Bar Charts	Top 20 calorie and fat-rich foods
🧮 3D Plot	Carbohydrate content vs category
📦 Boxen Plot	Calorie distribution across food categories

📜 License
MIT License
