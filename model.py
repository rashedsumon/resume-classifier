import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from data_loader import load_resume_data

def train_and_save_model():
    """
    Loads data, trains a machine learning pipeline, and saves it locally.
    """
    # 1. Fetch data using our data_loader module
    df = load_resume_data()
    
    X = df['Resume_Text']
    y = df['Category']
    
    # 2. Split data into Training and Testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training the AI Model Pipeline (TF-IDF + Logistic Regression)...")
    # 3. Create an ML Pipeline containing both vectorizer and classifier
    model_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    # 4. Train the model
    model_pipeline.fit(X_train, y_train)
    
    # 5. Evaluate the model
    predictions = model_pipeline.predict(X_test)
    print("\nModel Evaluation Report:\n")
    print(classification_report(y_test, predictions))
    
    # 6. Save the trained pipeline to disk
    joblib.dump(model_pipeline, 'resume_model.pkl')
    print("Model successfully saved as 'resume_model.pkl'")

if __name__ == "__main__":
    train_and_save_model()