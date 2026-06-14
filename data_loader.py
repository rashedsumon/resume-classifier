import os
import glob
import kagglehub
import pandas as pd

def load_resume_data():
    """
    Downloads the resume dataset from Kaggle using kagglehub
    and returns a cleaned Pandas DataFrame. Handles nested directory structures.
    """
    print("Downloading dataset from Kaggle...")
    # Download latest version
    path = kagglehub.dataset_download("snehaanbhawal/resume-dataset")
    print("Path to dataset files:", path)
    
    # FIX: Search recursively down the folder tree for ANY .csv file
    csv_files = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in the downloaded dataset path. Looked inside: {path}")
    
    # Choose the first matched CSV file
    target_csv = csv_files[0]
    print(f"Found CSV File at: {target_csv}")
    
    # Load the dataset
    df = pd.read_csv(target_csv)
    
    # Clean and match column syntax for snehaanbhawal/resume-dataset
    # This dataset uses 'Resume_str' for text content and 'Category' for target labels
    if 'Resume_str' in df.columns:
        df = df.rename(columns={'Resume_str': 'Resume_Text'})
    elif 'Resume' in df.columns:
        df = df.rename(columns={'Resume': 'Resume_Text'})
        
    # Ensure mandatory columns exist before subsetting
    if 'Category' not in df.columns or 'Resume_Text' not in df.columns:
        raise KeyError(f"Expected columns 'Category' and 'Resume_Text' not found. Available: {list(df.columns)}")
        
    return df[['Category', 'Resume_Text']]

if __name__ == "__main__":
    # Quick diagnostics test
    try:
        data = load_resume_data()
        print(f"\nSuccess! Dataset loaded successfully. Shape: {data.shape}")
        print(data.head(2))
    except Exception as e:
        print(f"Error occurred: {e}")