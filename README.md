# Salary-Predict

This project aims to provide a predictive model that should help HR teams make informed
decisions about compensation levels in an organization. Using sophisticated data analytics and
machine learning techniques, these models predict optimal compensation for existing employees
and potential compensation based on a wide variety of factors.

Procedures
To develop predictive models to assist HR teams in making informed decisions using a various
techniques and methods

System Architecture
The salary prediction model was developed and deployed using Docker containers to ensure
consistent and reproducible environments across different stages of the model lifecycle.

Data Preprocessing and Transformation
• Loading the data
First, we use the function to load the dataset from the specified path. This ensures
that the data is loaded only once, improving performance when dealing with large
data sets.
• Data Preprocessing
We modify the Data Frame to include only the relevant attributes specified in the layout.
This step involves filtering the data based on certain conditions and renaming the columns
for clarity.
A. Process Age Column: Created more meaningful categories of age values,
especially combining "45-54 years" with "55-64 years" and "over 45 years."
B. Clean Education Level: For simplicity, evenly arrange education levels into
broad groups.
C. Process Developer Types: Split a Dev Type column with semicolon-separated
values into binary columns.
D. Process Country Column: Group states as "other" if they appear to be below the
specified cutoff.
E. Process Years of Professional Coding: Change the “Years Code Pro” column
numerically, handling keywords such as "more than 50 years" and "less than 1
year".
F. Process Languages Worked: Split the “Language Have Worked” With column
into a separate binary column for each language, similar to the Dev Type object.
G. Process Platforms Worked: Split the” Platform Have Worked “With column into
two separate columns.
H. Process Tools and Technologies Worked: Split the “Tools Tech Have Worked”
with column into two separate columns.
I. Filter Salary: Filter rows with salaries over $300,000 to remove outliers

• Loading the preprocessed Data
 All of these processing steps combine together in a process, which makes pre-loaded
data and applies all changes to make a cleaned dataframe.

• Scoring Metrics
Defining a dictionary of scoring metrices to evaluate the model performance

• Loading the dataset
 Loaded the dataset for the model selection
 
• Splitting Data into Training and Testing Sets
Performed the splitting process in the dataframe into training (95%) and testing
(5%) sets. it separates features (X_train, X_test) from the target variable Salary
(y_train, y_test).

• Model Selection and Evaluation
 Defining and evaluating several machine learning models using cross validation
method
 The models used include:
• Decision Tree Regressor
• AdaBoost Regressor
• Bagging Regressor
• Random Forest Regressor
• Gradient Boosting Regressor

• Train and save the model
After training and testing all the models, Gradient Boosting regressor has better accuracy
when compared to other models. The best score of 0.6210. So, we are saving the model
for the future predictions.
