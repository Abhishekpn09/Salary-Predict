import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MaxAbsScaler
import joblib
from config import config

def load_model():
    return joblib.load(config.CKPT_PATH)

# Load the trained model
model = joblib.load('best_model.joblib')

model_columns = ['RemoteWork', 'EdLevel', 'YearsCodePro', 'Country', 'Age', 'DevType', 
                 'LanguageHaveWorkedWith', 'PlatformHaveWorkedWith', 'ToolsTechHaveWorkedWith',
                 'Pip', 'Vite', 'Bash/Shell (all shells)', 'Vercel', 'Maven (build tool)', 
                 'Make', 'Terraform', 'Webpack', 'Gradle', 'NuGet', 'Netlify', 'Digital Ocean', 
                 'Amazon Web Services (AWS)', 'Cloudflare', 
                 'Developer, front-end', 'Developer, mobile', 'TypeScript', 'Docker', 'C++', 
                 'Developer, desktop or enterprise applications', 'Rust', 'Kubernetes', 'Kotlin', 
                 'Engineering manager', 'Microsoft Azure', 'Homebrew', 'Heroku', 'JavaScript', 
                 'Developer, back-end', 'Python', 'Go', 'Yarn', 'Firebase', 'Ruby', 'SQL', 'C', 
                 'Java', 'PHP', 'PowerShell', 'C#', 'HTML/CSS', 'Google Cloud', 'Developer, full-stack', 'npm']

def get_user_input():  
    RemoteWork = input("\nOn-site/Remote (Fully remote, Hybrid (some remote, some in-person), Full in-person): ")
    EdLevel = input("\nEducation Level (Bachelor’s degree, Master’s degree, Post grad, Less than a Bachelors): ")
    YearsCodePro = float(input("\nYears of experience (0-46): "))
    Country = input(f"\nCountry {config.COUNTRY}: ")
    Age = input("\nAge (18-24 years old, 25-34 years old, 35-44 years old, Over 45): ")
    DevType = input(f"\nType of Developer {config.TYPES}: ")
    LanguageHaveWorkedWith = input(f"\nProgramming Language {config.LANGUAGE}: ")
    PlatformHaveWorkedWith = input(f"\nPlatform {config.PLATFORM}: ")
    ToolsTechHaveWorkedWith = input(f"\nTools {config.TOOLS}: ")

    return RemoteWork, EdLevel, YearsCodePro, Country, Age, DevType, LanguageHaveWorkedWith, PlatformHaveWorkedWith, ToolsTechHaveWorkedWith

def prepare_input_data(input_data):
    # Create a DataFrame from input data
    input_df = pd.DataFrame([input_data])

    # Ensure all required columns are present in the input DataFrame
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0  

    # Reorder the columns to match the training data
    input_df = input_df[model_columns]
    
    return input_df

# Get user input
RemoteWork, EdLevel, YearsCodePro, Country, Age, DevType, LanguageHaveWorkedWith, PlatformHaveWorkedWith, ToolsTechHaveWorkedWith = get_user_input()

# Prepare input data
input_data = {
    'RemoteWork': RemoteWork,
    'EdLevel': EdLevel,
    'YearsCodePro': YearsCodePro,
    'Country': Country,
    'Age': Age,
    'DevType': DevType,
    'LanguageHaveWorkedWith': LanguageHaveWorkedWith,
    'PlatformHaveWorkedWith': PlatformHaveWorkedWith,
    'ToolsTechHaveWorkedWith': ToolsTechHaveWorkedWith
}

prepared_input = prepare_input_data(input_data)
print("Prepared input data:")
print(prepared_input)

# Make prediction
pred = model.predict(prepared_input)
print("Prediction:", pred)
