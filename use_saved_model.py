#!/usr/bin/env python3
"""
Example: How to use a saved SVM model file
Run this script to load and use your saved model on new data
"""

import pickle
import pandas as pd
import numpy as np

def load_saved_model(model_filename):
    """Load a saved SVM model and all its components."""
    with open(model_filename, 'rb') as f:
        saved_data = pickle.load(f)
    
    model = saved_data['model']
    scaler = saved_data['scaler']
    label_encoder = saved_data['label_encoder']
    feature_columns = saved_data['feature_columns']
    timestamp = saved_data['timestamp']
    
    print(f"✅ Model loaded successfully!")
    print(f"📅 Saved on: {timestamp}")
    print(f"🔧 Features required: {feature_columns}")
    
    return model, scaler, label_encoder, feature_columns

def predict_new_data(model, scaler, label_encoder, feature_columns, new_data_file):
    """Make predictions on new data using the saved model."""
    
    # Load new data
    new_data = pd.read_csv(new_data_file)
    print(f"📊 New data shape: {new_data.shape}")
    
    # Check if all required features are present
    missing_features = [f for f in feature_columns if f not in new_data.columns]
    if missing_features:
        print(f"❌ Missing features: {missing_features}")
        return None
    
    # Extract features in the same order as training
    X_new = new_data[feature_columns].values
    
    # Apply the same scaling as during training
    X_new_scaled = scaler.transform(X_new)
    
    # Make predictions
    predictions_numeric = model.predict(X_new_scaled)
    
    # Convert back to original labels if label encoder was used
    if label_encoder:
        predictions_labels = label_encoder.inverse_transform(predictions_numeric)
    else:
        predictions_labels = predictions_numeric
    
    # Add predictions to the dataframe
    result_df = new_data.copy()
    result_df['Predicted_Label'] = predictions_labels
    result_df['Prediction_Numeric'] = predictions_numeric
    
    print(f"🔮 Predictions completed!")
    print(f"📈 Prediction distribution:")
    print(pd.Series(predictions_labels).value_counts())
    
    return result_df

# Example usage
if __name__ == "__main__":
    # Step 1: Load your saved model
    # Replace with your actual model filename
    model_filename = "svm_model_20251015_164500.pkl"  # Change this to your file
    
    try:
        model, scaler, label_encoder, feature_columns = load_saved_model(model_filename)
        
        # Step 2: Predict on new data
        # Replace with your new data file
        new_data_file = "sample_unseen_data.csv"  # Change this to your file
        
        results = predict_new_data(model, scaler, label_encoder, feature_columns, new_data_file)
        
        if results is not None:
            # Step 3: Save results
            output_filename = f"predictions_{model_filename.replace('.pkl', '.csv')}"
            results.to_csv(output_filename, index=False)
            print(f"💾 Results saved to: {output_filename}")
            
            # Display first few predictions
            print("\n📋 First 10 predictions:")
            print(results[['Predicted_Label'] + feature_columns].head(10))
            
    except FileNotFoundError:
        print(f"❌ Model file '{model_filename}' not found!")
        print("💡 Available model files:")
        import glob
        model_files = glob.glob("svm_model_*.pkl")
        for f in model_files:
            print(f"  - {f}")