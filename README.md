# 🎓Student Academic Performance Prediction: Regression Model🎓

## 📚Project Overview

This project focuses on developing a Machine Learning regression model to predict the final exam_score of students based on a comprehensive set of variables, including academic habits, lifestyle factors, mental health, and demographics. The primary objective is to quantify the impact of non-academic factors—such as sleep, diet quality, and social media usage—on educational outcomes.

## 📚Business Question & Hypothesis

- Main Question: What is the quantifiable impact of lifestyle and study habits on a student's academic performance?
- Hypothesis (H1): It is possible to build a regression model that predicts the final exam score with an $R^2$ (R-squared) significantly greater than a naive model, aiming to achieve an $R^2 > 0.60$.

## 📚Technical Stack

- Language: Python
- Libraries: Pandas, NumPy, Matplotlib, Seaborn
- Machine Learning: Scikit-learn (LinearRegression, RandomForestRegressor, GradientBoostingRegressor, Pipeline, ColumnTransformer, RandomizedSearchCV)

## 📚Data Description

The simulated dataset contains 1,000 synthetic records covering 15 features related to student life:

- Target Variable: exam_score (float, final grade).
- Key Predictors: study_hours_per_day, sleep_hours, diet_quality, exercise_frequency, social_media_hours, mental_health_rating, and internet_quality.

## 📚Data Preprocessing & Exploratory Data Analysis (EDA)

The notebook followed a robust data preparation and exploration strategy:

### 1. Data Cleaning & Feature Engineering

- Missing Values: The 91 missing values in the parental_education_level column were imputed with the category 'Unknown' to ensure the entire dataset could be utilized for modeling without introducing bias through deletion or mean imputation.
- Feature Encoding: Categorical features were prepared for modeling using a ColumnTransformer to ensure data integrity:

OrdinalEncoder was applied to ranked data (diet_quality, parental_education_level, internet_quality), respecting the defined order of categories (e.g., 'Poor' to 'Good').
OneHotEncoder was applied to nominal data (gender, part_time_job, country_of_origin).

### 2. Key Insights from EDA

The correlation matrix and scatter plots revealed powerful relationships between student habits and their final score:

- Strongest Predictor: A strong positive linear correlation was confirmed between exam_score and study_hours_per_day (correlation coefficient not shown here but visually evident).
- Impact of Wellness: Students who maintain a "Good" or "Fair" diet quality and who exercise 5-6 times per week consistently achieved higher average exam scores.
- Technology & Lifestyle: Time spent on social media and Netflix showed a weak negative correlation with exam performance. Conversely, better internet_quality was associated with a higher average exam_score.
- Sleep: Sleep hours showed a weak positive correlation with the final score, suggesting that while rest is beneficial, it is less impactful on the score than dedicated study time.

## 📚Machine Learning Pipeline (Comparative Regression)

Three regression models (Linear Regression, Random Forest, and Gradient Boosting) were trained and compared. The ColumnTransformer was used to ensure the features were processed correctly for each model.

- Model Training: All models were trained and evaluated using 5-Fold Cross-Validation on the training set.
- Hyperparameter Optimization: A RandomizedSearchCV approach was used to tune the Gradient Boosting Regressor to maximize predictive performance.
- Winner: The Linear Regression model provided the best balance of simplicity and accuracy on the final test set.

## 📚Model Performance & Conclusion

The Linear Regression model fue el de mejor rendimiento en el conjunto de prueba, superando a los modelos Random Forest y Gradient Boosting (incluso después de su optimización). El modelo alcanzó una alta capacidad predictiva con las siguientes métricas:

- R-squared ($R^2$): 0.90 (Mide la proporción de la varianza explicada, superando ampliamente el objetivo de 0.60).
- Mean Absolute Error (MAE): 4.15 (El error promedio absoluto en la predicción de la nota fue de solo 4.15 puntos).
- Mean Squared Error (MSE): 25.83.

## 📚Final Conclusion:

Based on the achieved $R^2$ of 0.90, the model successfully confirmed the alternate hypothesis ($H_1$). The primary driver of predictive power is the time dedicated to studying, reinforcing that focused study habits remain the most critical factor for academic success, even outweighing lifestyle factors like sleep or mental health rating in a multivariate model.

## 📚Future Work

- Exploration of alternative models such as Support Vector Regression.
- Analysis of feature importance to formally rank the contribution of each variable (e.g., how much stronger is study_hours_per_day compared to exercise_frequency).
- Creation of an ensemble model combining the stability of Linear Regression with the finer predictive power of the optimized Gradient Boosting Regressor.

# 🎓 Modelo de Predicción de Rendimiento Académico Estudiantil: Regresión🎓

## 📚Resumen del Proyecto

Este proyecto se centra en el desarrollo de un modelo de regresión de Machine Learning para predecir la exam_score (nota final del examen) de los estudiantes basándose en un conjunto integral de variables, que incluyen hábitos académicos, factores de estilo de vida, salud mental y demografía. El objetivo principal es cuantificar el impacto de los factores no académicos, como las horas de sueño, la calidad de la dieta y el uso de redes sociales, en los resultados educativos.

## 📚Pregunta de Negocio e Hipótesis

- Pregunta Principal: ¿Cuál es el impacto cuantificable del estilo de vida y los hábitos de estudio en el rendimiento académico de un estudiante?
- Hipótesis (H1): Es posible construir un modelo de regresión que prediga la nota final del examen con un $R^2$ (R-cuadrado) significativamente superior a un modelo ingenuo, buscando lograr un $R^2 > 0.60$.

## 📚Stack Técnico

- Lenguaje: Python
- Librerías: Pandas, NumPy, Matplotlib, Seaborn
- Machine Learning: Scikit-learn (LinearRegression, RandomForestRegressor, GradientBoostingRegressor, Pipeline, ColumnTransformer, RandomizedSearchCV)

## 📚Descripción de los Datos

El conjunto de datos simulado contiene 1,000 registros sintéticos que cubren 15 características relacionadas con la vida estudiantil:

- Variable Objetivo: exam_score (float, nota final).
- Predictoras Clave: study_hours_per_day (horas de estudio por día), sleep_hours (horas de sueño), diet_quality (calidad de la dieta), exercise_frequency (frecuencia de ejercicio), social_media_hours (horas de redes sociales), mental_health_rating (calificación de salud mental) y internet_quality (calidad de internet).

## 📚Preprocesamiento de Datos y Análisis Exploratorio (EDA)

El notebook siguió una sólida estrategia de preparación y exploración de datos:

### 1. Limpieza de Datos e Ingeniería de Características

- Valores Faltantes: Los 91 valores faltantes en la columna parental_education_level se imputaron con la categoría 'Unknown' (Desconocido) para garantizar que todo el conjunto de datos pudiera utilizarse para el modelado sin introducir sesgos.
- Codificación de Características: Las características categóricas se prepararon para el modelado utilizando un ColumnTransformer:

Se aplicó OrdinalEncoder a los datos clasificados (diet_quality, parental_education_level, internet_quality), respetando el orden definido de las categorías (por ejemplo, 'Poor' a 'Good').

Se aplicó OneHotEncoder a los datos nominales (gender, part_time_job, country_of_origin).

### 2. Hallazgos Clave del EDA

La matriz de correlación y los gráficos de dispersión revelaron poderosas relaciones entre los hábitos de los estudiantes y su nota final:

- Predictor más Fuerte: Se confirmó una fuerte correlación lineal positiva entre exam_score y study_hours_per_day (horas de estudio por día).
- Impacto del Bienestar: Los estudiantes que mantienen una calidad de dieta "Buena" o "Aceptable" y que hacen ejercicio 5-6 veces por semana lograron constantemente puntuaciones de examen promedio más altas.
- Tecnología y Estilo de Vida: El tiempo dedicado a las redes sociales y Netflix mostró una correlación negativa débil con el rendimiento. Por el contrario, una mejor internet_quality se asoció con una exam_score promedio más alta.
- Sueño: Las horas de sueño mostraron una correlación positiva débil con la nota final, lo que sugiere que, si bien el descanso es beneficioso, tiene un impacto menor en la calificación que el tiempo de estudio dedicado.

## 📚Pipeline de Machine Learning (Regresión Comparativa)

Se entrenaron y compararon tres modelos de regresión (Regresión Lineal, Random Forest y Gradient Boosting).

- Entrenamiento del Modelo: Todos los modelos se entrenaron y evaluaron utilizando Validación Cruzada de 5 Pliegues en el conjunto de entrenamiento.
- Optimización de Hiperparámetros: Se utilizó un enfoque de RandomizedSearchCV para ajustar el Gradient Boosting Regressor y maximizar el rendimiento predictivo.
- Ganador: El modelo de Regresión Lineal proporcionó el mejor equilibrio entre simplicidad y precisión en el conjunto de prueba final.

## 📚Rendimiento del Modelo y Conclusión

El modelo de Regresión Lineal fue el de mejor rendimiento en el conjunto de prueba, superando a los modelos Random Forest y Gradient Boosting (incluso después de su optimización). El modelo alcanzó una alta capacidad predictiva con las siguientes métricas:

- R-squared ($R^2$): 0.90 (Mide la proporción de la varianza explicada, superando ampliamente el objetivo de 0.60).
- Mean Absolute Error (MAE): 4.15 (El error promedio absoluto en la predicción de la nota fue de solo 4.15 puntos).
- Mean Squared Error (MSE): 25.83.

## 📚Conclusión Final:

Basado en el $R^2$ logrado de 0.90, el modelo confirmó exitosamente la hipótesis alternativa ($H_1$). El principal impulsor del poder predictivo es el tiempo dedicado al estudio, lo que refuerza que los hábitos de estudio enfocados siguen siendo el factor más crítico para el éxito académico, incluso superando a los factores de estilo de vida como el sueño o la calificación de salud mental en un modelo multivariado.

## 📚Trabajo Futuro

- Exploración de modelos alternativos como Support Vector Regression.
- Análisis de la importancia de las características para clasificar formalmente la contribución de cada variable (por ejemplo, qué tan fuerte es study_hours_per_day en comparación con exercise_frequency).
- Creación de un modelo de ensamble que combine la estabilidad de la Regresión Lineal con el poder predictivo más fino del Gradient Boosting Regressor optimizado.
