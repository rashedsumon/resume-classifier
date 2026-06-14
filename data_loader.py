import os
import glob
import kagglehub
import pandas as pd

def load_resume_data():
    """
    Downloads the resume dataset from Kaggle using kagglehub
    and returns a cleaned Pandas DataFrame.
    """
    print("Downloading dataset from Kaggle...")
    # Download latest version
    path = kagglehub.dataset_download("snehaanbhawal/resume-dataset")
    print("Path to dataset files:", path)
    
    # Locate the CSV file inside the downloaded directory
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the downloaded dataset path.")
    
    # Load the dataset (usually named Resume.csv)
    df = pd.read_csv(csv_files[0])
    
    # Ensure standard column naming based on this specific dataset
    # Expected columns: 'Category' and 'Resume_str' or 'Resume'
    if 'Resume_str' in df.columns:
        df = df.rename(columns={'Resume_str': 'Resume_Text'})
    elif 'Resume' in df.columns:
        df = df.rename(columns={'Resume': 'Resume_Text'})
        
    return df[['Category', 'Resume_Text']]

if __name__ == "__main__":
    # Test script locally
    data = load_resume_data()
    print(f"Dataset loaded successfully! Shape: {data.shape}")
    print(data.head(2))