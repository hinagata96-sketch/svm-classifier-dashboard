# === SVM Streamlit Dashboard ===
# Run with: streamlit run svm.py

# === Imports ===
import streamlit as st
import io
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# === Page Configuration ===
st.set_page_config(
    page_title="SVM Classifier Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Initialize Session State ===
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'train_data' not in st.session_state:
    st.session_state.train_data = None
if 'best_params' not in st.session_state:
    st.session_state.best_params = None
if 'label_encoder' not in st.session_state:
    st.session_state.label_encoder = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'unseen_results' not in st.session_state:
    st.session_state.unseen_results = None
if 'grid_search_used' not in st.session_state:
    st.session_state.grid_search_used = False

# === Helper Functions ===
def is_onehot(df, cols):
    """Return True if columns are one-hot encoded (0/1, one 1 per row)."""
    subset = df[list(cols)]
    if not all(subset.dropna().map(lambda x: x in (0, 1)).all()):
        return False
    s = subset.sum(axis=1)
    return ((s == 1).all() and subset.shape[1] > 1)

def save_model_and_params(model, scaler, label_encoder, best_params, feature_cols):
    """Save the trained model and parameters to files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_filename = f"svm_model_{timestamp}.pkl"
    with open(model_filename, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'feature_columns': feature_cols,
            'timestamp': timestamp
        }, f)
    
    # Save parameters
    params_filename = f"svm_params_{timestamp}.json"
    with open(params_filename, 'w') as f:
        json.dump({
            'best_params': best_params,
            'feature_columns': feature_cols,
            'timestamp': timestamp
        }, f, indent=2)
    
    return model_filename, params_filename

def load_model_files():
    """Load available model files."""
    import glob
    model_files = glob.glob("svm_model_*.pkl")
    return sorted(model_files, reverse=True)  # Most recent first

# === Main App ===
def main():
    st.title("🤖 SVM Classifier Dashboard")
    st.write("Upload a CSV file and train an SVM model with different kernels")
    
    # === Sidebar for Model Configuration ===
    st.sidebar.header("Model Configuration")
    
    # File Upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your dataset in CSV format"
    )
    
    # Handle file upload
    if uploaded_file is not None:
        try:
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success(f"✅ Data Loaded: {uploaded_file.name}")
            st.write(f"📐 Shape: {st.session_state.df.shape}")
            
            # Display dataset preview
            with st.expander("Dataset Preview", expanded=True):
                st.dataframe(st.session_state.df.head())
                
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return
    
    # Only show configuration options if data is loaded
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Feature and target selection
        st.sidebar.subheader("Column Selection")
        
        feature_cols = st.sidebar.multiselect(
            "Select Feature Columns",
            options=df.columns.tolist(),
            help="Choose the input features for training"
        )
        
        target_cols = st.sidebar.multiselect(
            "Select Target Column(s)",
            options=df.columns.tolist(),
            help="Choose the target variable(s)"
        )
        
        # Model parameters
        st.sidebar.subheader("Model Parameters")
        
        # Hyperparameter tuning option
        use_grid_search = st.sidebar.checkbox(
            "🔍 Use Grid Search for Best Parameters",
            value=True,
            help="Find the best hyperparameters automatically"
        )
        
        # Kernel selection - always available
        st.sidebar.write("**Kernel Selection:**")
        kernel_option = st.sidebar.radio(
            "Choose Kernel Strategy",
            options=["Auto (Grid Search)", "Manual Selection"],
            index=0 if use_grid_search else 1,
            help="Choose how to select the kernel type"
        )
        
        if kernel_option == "Manual Selection" or not use_grid_search:
            kernel_type = st.sidebar.selectbox(
                "SVM Kernel",
                options=['linear', 'rbf', 'poly', 'sigmoid'],
                index=1,  # Default to RBF
                help="Choose the kernel function for SVM"
            )
            
            # Additional parameters for manual selection
            if kernel_type in ['rbf', 'poly', 'sigmoid']:
                manual_c = st.sidebar.selectbox(
                    "C Parameter",
                    options=[0.1, 1, 10, 100],
                    index=1,  # Default to C=1
                    help="Regularization parameter"
                )
                
                manual_gamma = st.sidebar.selectbox(
                    "Gamma Parameter",
                    options=['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    index=0,  # Default to 'scale'
                    help="Kernel coefficient"
                )
            else:
                manual_c = st.sidebar.selectbox(
                    "C Parameter",
                    options=[0.1, 1, 10, 100],
                    index=1,  # Default to C=1
                    help="Regularization parameter"
                )
                manual_gamma = None
            
            if kernel_type == 'poly':
                manual_degree = st.sidebar.selectbox(
                    "Degree (Polynomial only)",
                    options=[2, 3, 4, 5],
                    index=1,  # Default to degree=3
                    help="Degree of polynomial kernel"
                )
            else:
                manual_degree = None
        else:
            # Grid search mode
            kernel_type = None
            manual_c = None
            manual_gamma = None
            manual_degree = None
            
            st.sidebar.info("Grid Search will test multiple kernels and find the best combination of parameters")
        
        train_ratio = st.sidebar.slider(
            "Training Data Ratio",
            min_value=0.5,
            max_value=0.9,
            value=0.7,
            step=0.05,
            help="Proportion of data used for training"
        )
        
        # Training section
        st.sidebar.subheader("Actions")
        
        train_clicked = st.sidebar.button(
            "🚀 Train SVM Model",
            type="primary",
            use_container_width=True
        )
        
        test_clicked = st.sidebar.button(
            "🧪 Test Model",
            use_container_width=True
        )
        
        # Unseen data testing section
        st.sidebar.subheader("Test on Unseen Data")
        
        unseen_file = st.sidebar.file_uploader(
            "Upload Unseen Test Data",
            type="csv",
            key="unseen_data",
            help="Upload a CSV file with the same features for testing"
        )
        
        if unseen_file is not None:
            test_unseen_clicked = st.sidebar.button(
                "🔮 Test on Unseen Data",
                use_container_width=True,
                type="secondary"
            )
        else:
            test_unseen_clicked = False
        
        # Model saving/loading section
        st.sidebar.subheader("Model Management")
        
        if st.session_state.model is not None:
            save_model_clicked = st.sidebar.button(
                "💾 Save Best Model",
                use_container_width=True
            )
        else:
            save_model_clicked = False
        
        # Load existing models
        model_files = load_model_files()
        if model_files:
            selected_model = st.sidebar.selectbox(
                "Load Saved Model",
                options=["None"] + model_files,
                help="Load a previously saved model"
            )
            
            if selected_model != "None":
                load_model_clicked = st.sidebar.button(
                    "📂 Load Selected Model",
                    use_container_width=True
                )
            else:
                load_model_clicked = False
        else:
            # Show message when no saved models are available
            st.sidebar.info("💡 No saved models found. Train and save a model first!")
            selected_model = "None"
            load_model_clicked = False
        
        # === Main Content Area ===
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Training logic with Grid Search
            if train_clicked:
                if not feature_cols or not target_cols:
                    st.error("Please select at least one feature and target column.")
                    return
                
                try:
                    with st.spinner("Training SVM model... This may take a few minutes."):
                        # Handle one-hot or single label
                        le = None
                        if is_onehot(df, target_cols):
                            y = df[target_cols].idxmax(axis=1)
                            st.info(f"Detected one-hot encoding for target. Converted to labels: {list(df[target_cols].columns)}")
                        elif len(target_cols) == 1:
                            y = df[target_cols[0]]
                        else:
                            st.error("Multiple targets selected but not one-hot encoded. Please select valid target columns.")
                            return
                        
                        # Encode y if not integer
                        if not np.issubdtype(y.dtype, np.integer):
                            le = LabelEncoder()
                            y = le.fit_transform(y)
                            st.info(f"Encoded target labels: {dict(zip(le.classes_, le.transform(le.classes_)))}")
                        else:
                            y = y.values
                        
                        X = df[feature_cols].values
                        
                        # Normalize
                        scaler = MinMaxScaler()
                        X_scaled = scaler.fit_transform(X)
                        
                        # Split according to train_ratio
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_scaled, y, test_size=1-train_ratio, random_state=42, stratify=y
                        )
                        X_train, X_val, y_train, y_val = train_test_split(
                            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
                        )
                        
                        # Train model with or without grid search
                        if kernel_option == "Auto (Grid Search)" and use_grid_search:
                            st.info("🔍 Performing Grid Search to find best parameters...")
                            
                            # Define parameter grid
                            param_grid = [
                                {
                                    'kernel': ['linear'],
                                    'C': [0.1, 1, 10, 100]
                                },
                                {
                                    'kernel': ['rbf'],
                                    'C': [0.1, 1, 10, 100],
                                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
                                },
                                {
                                    'kernel': ['poly'],
                                    'C': [0.1, 1, 10, 100],
                                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                                    'degree': [2, 3, 4]
                                }
                            ]
                            
                            # Perform grid search
                            grid_search = GridSearchCV(
                                SVC(),
                                param_grid,
                                cv=5,
                                scoring='accuracy',
                                n_jobs=-1,
                                verbose=0
                            )
                            
                            grid_search.fit(X_train, y_train)
                            model = grid_search.best_estimator_
                            best_params = grid_search.best_params_
                            
                            st.success(f"✅ Best parameters found: {best_params}")
                            st.info(f"🎯 Best CV Score: {grid_search.best_score_:.3f}")
                            
                        else:
                            # Manual kernel selection
                            st.info(f"🔧 Training with manual kernel selection: {kernel_type}")
                            
                            # Build parameters manually
                            model_params = {'kernel': kernel_type}
                            if manual_c is not None:
                                model_params['C'] = manual_c
                            if manual_gamma is not None:
                                model_params['gamma'] = manual_gamma
                            if manual_degree is not None:
                                model_params['degree'] = manual_degree
                            
                            model = SVC(**model_params)
                            model.fit(X_train, y_train)
                            best_params = model_params
                        
                        # Store in session state
                        st.session_state.model = model
                        st.session_state.scaler = scaler
                        st.session_state.label_encoder = le
                        st.session_state.feature_columns = feature_cols
                        st.session_state.best_params = best_params
                        st.session_state.grid_search_used = (kernel_option == "Auto (Grid Search)" and use_grid_search)
                        st.session_state.train_data = {
                            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
                            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
                            'feature_cols': feature_cols
                        }
                        
                        # Calculate accuracies
                        train_acc = accuracy_score(y_train, model.predict(X_train))
                        val_acc = accuracy_score(y_val, model.predict(X_val))
                        
                        if kernel_option == "Auto (Grid Search)" and use_grid_search:
                            st.success(f"✅ Model trained with Grid Search!")
                        else:
                            st.success(f"✅ Model trained with manual selection!")
                        
                        # Display metrics
                        if kernel_option == "Auto (Grid Search)" and use_grid_search:
                            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                            with metrics_col1:
                                st.metric("Training Accuracy", f"{train_acc:.3f}")
                            with metrics_col2:
                                st.metric("Validation Accuracy", f"{val_acc:.3f}")
                            with metrics_col3:
                                st.metric("CV Score", f"{grid_search.best_score_:.3f}")
                        else:
                            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                            with metrics_col1:
                                st.metric("Training Accuracy", f"{train_acc:.3f}")
                            with metrics_col2:
                                st.metric("Validation Accuracy", f"{val_acc:.3f}")
                            with metrics_col3:
                                st.metric("Selected Kernel", best_params['kernel'])
                        
                        # Display best parameters
                        st.subheader("🏆 Model Parameters")
                        param_cols = st.columns(len(best_params))
                        for i, (param, value) in enumerate(best_params.items()):
                            with param_cols[i]:
                                st.metric(param.upper(), str(value))
                        
                        # Show comparison if grid search was used
                        if kernel_option == "Auto (Grid Search)" and use_grid_search:
                            with st.expander("📊 Grid Search Results Comparison", expanded=False):
                                # Show top 5 parameter combinations
                                results_df = pd.DataFrame(grid_search.cv_results_)
                                top_results = results_df.nlargest(5, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
                                
                                st.write("**Top 5 Parameter Combinations:**")
                                for idx, row in top_results.iterrows():
                                    score = row['mean_test_score']
                                    std = row['std_test_score']
                                    params = row['params']
                                    
                                    if idx == top_results.index[0]:  # Best result
                                        st.success(f"🥇 **Best:** {params} → Accuracy: {score:.3f} (±{std:.3f})")
                                    else:
                                        st.info(f"📊 {params} → Accuracy: {score:.3f} (±{std:.3f})")
                        
                        # Add manual override option even after grid search
                        if kernel_option == "Auto (Grid Search)" and use_grid_search:
                            st.subheader("🔄 Want to Try a Different Kernel?")
                            st.info("💡 **Tip:** You can switch to 'Manual Selection' in the sidebar to try specific kernels and compare with the grid search results!")
                        
                        # Plot if at least 2 features
                        if X.shape[1] >= 2:
                            st.subheader("📊 Decision Boundary Visualization")
                            fx, fy = 0, 1
                            X_2D = X_scaled[:, [fx, fy]]
                            
                            # Create 2D model with same parameters
                            model_2D = SVC(**best_params)
                            model_2D.fit(X_2D, y)
                            
                            x_min, x_max = X_2D[:, 0].min() - 0.1, X_2D[:, 0].max() + 0.1
                            y_min, y_max = X_2D[:, 1].min() - 0.1, X_2D[:, 1].max() + 0.1
                            xx, yy = np.meshgrid(
                                np.linspace(x_min, x_max, 100),
                                np.linspace(y_min, y_max, 100)
                            )
                            Z = model_2D.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            contour = ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
                            scatter = ax.scatter(X_2D[:, 0], X_2D[:, 1], c=y, cmap='viridis', edgecolors='k')
                            
                            if kernel_option == "Auto (Grid Search)" and use_grid_search:
                                title = f"SVM Decision Boundary (Grid Search Best: {best_params['kernel']})"
                            else:
                                title = f"SVM Decision Boundary (Manual: {best_params['kernel']})"
                            
                            ax.set_title(title)
                            ax.set_xlabel(feature_cols[fx])
                            ax.set_ylabel(feature_cols[fy])
                            ax.grid(True, alpha=0.3)
                            plt.colorbar(scatter, ax=ax)
                            st.pyplot(fig)
                        else:
                            st.info("Decision boundary plot available for datasets with ≥2 features only.")
                
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")
                    st.exception(e)
            
            # Testing logic on validation/test set
            if test_clicked:
                if st.session_state.model is None or st.session_state.train_data is None:
                    st.error("❌ Train the model first.")
                else:
                    try:
                        model = st.session_state.model
                        train_data = st.session_state.train_data
                        
                        y_pred = model.predict(train_data['X_test'])
                        test_acc = accuracy_score(train_data['y_test'], y_pred)
                        
                        st.success("🧪 Model Testing Complete")
                        st.metric("Test Accuracy", f"{test_acc:.3f}")
                        
                        # Detailed classification report
                        st.subheader("📊 Detailed Test Results")
                        
                        # Classification report
                        if st.session_state.label_encoder:
                            target_names = st.session_state.label_encoder.classes_
                        else:
                            target_names = None
                        
                        report = classification_report(
                            train_data['y_test'], 
                            y_pred, 
                            target_names=target_names,
                            output_dict=True
                        )
                        
                        # Display metrics in a nice format
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Per-Class Metrics:**")
                            if target_names is not None:
                                for class_name in target_names:
                                    if str(class_name) in report:
                                        metrics = report[str(class_name)]
                                        st.write(f"**{class_name}:**")
                                        st.write(f"  - Precision: {metrics['precision']:.3f}")
                                        st.write(f"  - Recall: {metrics['recall']:.3f}")
                                        st.write(f"  - F1-Score: {metrics['f1-score']:.3f}")
                        
                        with col2:
                            st.write("**Overall Metrics:**")
                            st.write(f"**Accuracy:** {report['accuracy']:.3f}")
                            st.write(f"**Macro Avg F1:** {report['macro avg']['f1-score']:.3f}")
                            st.write(f"**Weighted Avg F1:** {report['weighted avg']['f1-score']:.3f}")
                        
                        # Confusion Matrix
                        st.subheader("🔄 Confusion Matrix")
                        cm = confusion_matrix(train_data['y_test'], y_pred)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
                        ax.figure.colorbar(im, ax=ax)
                        
                        if target_names is not None:
                            tick_marks = np.arange(len(target_names))
                            ax.set_xticks(tick_marks)
                            ax.set_yticks(tick_marks)
                            ax.set_xticklabels(target_names)
                            ax.set_yticklabels(target_names)
                        
                        # Add text annotations
                        thresh = cm.max() / 2.
                        for i in range(cm.shape[0]):
                            for j in range(cm.shape[1]):
                                ax.text(j, i, format(cm[i, j], 'd'),
                                       ha="center", va="center",
                                       color="white" if cm[i, j] > thresh else "black")
                        
                        ax.set_ylabel('True Label')
                        ax.set_xlabel('Predicted Label')
                        ax.set_title('Confusion Matrix')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                    except Exception as e:
                        st.error(f"Error during testing: {str(e)}")
                        st.exception(e)
            
            # Save model functionality
            if save_model_clicked:
                try:
                    model_file, params_file = save_model_and_params(
                        st.session_state.model,
                        st.session_state.scaler,
                        st.session_state.label_encoder,
                        st.session_state.best_params,
                        st.session_state.feature_columns
                    )
                    st.success(f"✅ Model saved successfully!")
                    st.info(f"📁 Model file: {model_file}")
                    st.info(f"📁 Parameters file: {params_file}")
                except Exception as e:
                    st.error(f"Error saving model: {str(e)}")
            
            # Load model functionality
            if load_model_clicked:
                try:
                    with open(selected_model, 'rb') as f:
                        saved_data = pickle.load(f)
                    
                    st.session_state.model = saved_data['model']
                    st.session_state.scaler = saved_data['scaler']
                    st.session_state.label_encoder = saved_data['label_encoder']
                    st.session_state.feature_columns = saved_data['feature_columns']
                    
                    st.success(f"✅ Model loaded successfully from {selected_model}")
                    st.info(f"📊 Model features: {saved_data['feature_columns']}")
                    st.info(f"⏰ Saved on: {saved_data['timestamp']}")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")
            
            # Test on unseen data
            if test_unseen_clicked:
                if st.session_state.model is None:
                    st.error("❌ No trained model available. Train or load a model first.")
                else:
                    try:
                        # Load unseen data
                        unseen_df = pd.read_csv(unseen_file)
                        st.success(f"✅ Unseen data loaded: {unseen_file.name}")
                        st.write(f"📐 Unseen data shape: {unseen_df.shape}")
                        
                        # Check if required features are present
                        required_features = st.session_state.feature_columns
                        missing_features = [f for f in required_features if f not in unseen_df.columns]
                        
                        if missing_features:
                            st.error(f"❌ Missing required features: {missing_features}")
                            st.info(f"Required features: {required_features}")
                            st.info(f"Available features: {list(unseen_df.columns)}")
                            return
                        
                        # Prepare unseen data
                        X_unseen = unseen_df[required_features].values
                        X_unseen_scaled = st.session_state.scaler.transform(X_unseen)
                        
                        # Make predictions
                        y_pred_unseen = st.session_state.model.predict(X_unseen_scaled)
                        
                        # If we have a label encoder, inverse transform predictions
                        if st.session_state.label_encoder:
                            y_pred_labels = st.session_state.label_encoder.inverse_transform(y_pred_unseen)
                        else:
                            y_pred_labels = y_pred_unseen
                        
                        st.success("🔮 Predictions on unseen data completed!")
                        
                        # Display predictions
                        st.subheader("📊 Unseen Data Predictions")
                        
                        # Create results dataframe
                        results_df = unseen_df.copy()
                        results_df['Predicted_Label'] = y_pred_labels
                        results_df['Prediction_Numeric'] = y_pred_unseen
                        
                        # Store results in session state
                        st.session_state.unseen_results = results_df
                        
                        # Display results
                        st.dataframe(results_df.head(20))
                        
                        if len(results_df) > 20:
                            st.info(f"Showing first 20 rows. Total predictions: {len(results_df)}")
                        
                        # Prediction distribution
                        st.subheader("📈 Prediction Distribution")
                        pred_counts = pd.Series(y_pred_labels).value_counts()
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        pred_counts.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
                        ax.set_title('Distribution of Predictions on Unseen Data')
                        ax.set_xlabel('Predicted Class')
                        ax.set_ylabel('Count')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions CSV",
                            data=csv,
                            file_name=f"unseen_data_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Error testing on unseen data: {str(e)}")
                        st.exception(e)
        
        with col2:
            # Model info panel
            if st.session_state.model is not None:
                st.subheader("🤖 Model Information")
                
                if st.session_state.best_params:
                    st.write("**Current Parameters:**")
                    for param, value in st.session_state.best_params.items():
                        st.write(f"  - {param}: {value}")
                    
                    # Show whether this was from grid search or manual selection
                    if 'grid_search_used' in st.session_state and st.session_state.grid_search_used:
                        st.success("🔍 **Found by Grid Search**")
                        st.caption("These are the optimal parameters discovered automatically")
                    else:
                        st.info("🔧 **Manual Selection**")
                        st.caption("These parameters were chosen manually")
                    
                    # Add parameter explanations
                    with st.expander("🔍 Parameter Explanations", expanded=False):
                        params = st.session_state.best_params
                        
                        if 'kernel' in params:
                            st.write("**🔧 Kernel:**")
                            kernel = params['kernel']
                            if kernel == 'linear':
                                st.write("- **Linear**: Creates straight decision boundaries. Best for linearly separable data.")
                            elif kernel == 'rbf':
                                st.write("- **RBF (Radial Basis Function)**: Creates curved, circular decision boundaries. Good for complex, non-linear data.")
                            elif kernel == 'poly':
                                st.write("- **Polynomial**: Creates curved decision boundaries using polynomial functions. Good for moderately complex patterns.")
                            elif kernel == 'sigmoid':
                                st.write("- **Sigmoid**: Creates S-shaped decision boundaries. Similar to neural networks.")
                        
                        if 'C' in params:
                            st.write("**⚖️ C (Regularization):**")
                            c_val = params['C']
                            st.write(f"- **Value: {c_val}** - Controls the trade-off between smooth decision boundary and classifying training points correctly.")
                            if c_val < 1:
                                st.write("- **Low C**: More tolerant of errors, smoother decision boundary (less overfitting)")
                            elif c_val == 1:
                                st.write("- **Medium C**: Balanced approach between accuracy and generalization")
                            else:
                                st.write("- **High C**: Less tolerant of errors, more complex decision boundary (risk of overfitting)")
                        
                        if 'gamma' in params:
                            st.write("**🎯 Gamma (Kernel Coefficient):**")
                            gamma_val = params['gamma']
                            st.write(f"- **Value: {gamma_val}** - Defines how far the influence of a single training example reaches.")
                            if isinstance(gamma_val, str):
                                if gamma_val == 'scale':
                                    st.write("- **Scale**: Automatically calculated as 1/(n_features × X.var())")
                                elif gamma_val == 'auto':
                                    st.write("- **Auto**: Automatically calculated as 1/n_features")
                            else:
                                if gamma_val < 0.1:
                                    st.write("- **Low Gamma**: Far-reaching influence, smoother decision boundary")
                                elif gamma_val <= 1:
                                    st.write("- **Medium Gamma**: Moderate influence, balanced complexity")
                                else:
                                    st.write("- **High Gamma**: Close influence, more complex decision boundary")
                        
                        if 'degree' in params:
                            st.write("**📐 Degree (Polynomial only):**")
                            degree_val = params['degree']
                            st.write(f"- **Value: {degree_val}** - The degree of the polynomial kernel function.")
                            if degree_val == 2:
                                st.write("- **Degree 2**: Quadratic relationships (parabolic curves)")
                            elif degree_val == 3:
                                st.write("- **Degree 3**: Cubic relationships (S-shaped curves)")
                            elif degree_val >= 4:
                                st.write("- **Higher Degree**: More complex polynomial relationships")
                        
                        st.write("---")
                        st.write("**📊 Your Model Summary:**")
                        if 'kernel' in params and params['kernel'] == 'poly':
                            st.write("Your model uses a **polynomial kernel** which means:")
                            st.write("✅ It can capture curved, non-linear patterns in your data")
                            st.write("✅ It's more flexible than linear models")
                            st.write("✅ Good for data where classes are separated by curved boundaries")
                            
                            if 'degree' in params and params['degree'] == 3:
                                st.write("✅ Degree 3 allows for S-shaped and cubic decision boundaries")
                            
                            if 'C' in params:
                                c_val = params['C']
                                if c_val == 1:
                                    st.write("✅ Balanced regularization - good for most datasets")
                            
                            if 'gamma' in params and params['gamma'] == 1:
                                st.write("✅ Medium gamma - balanced complexity")
                
                if st.session_state.feature_columns:
                    st.write(f"**Features ({len(st.session_state.feature_columns)}):**")
                    for i, feature in enumerate(st.session_state.feature_columns, 1):
                        st.write(f"  {i}. {feature}")
                
                if st.session_state.train_data:
                    train_data = st.session_state.train_data
                    st.write("**Data Split:**")
                    st.write(f"  - Training: {len(train_data['y_train'])} samples")
                    st.write(f"  - Validation: {len(train_data['y_val'])} samples")
                    st.write(f"  - Test: {len(train_data['y_test'])} samples")
                
                if st.session_state.label_encoder:
                    st.write("**Label Encoding:**")
                    for i, label in enumerate(st.session_state.label_encoder.classes_):
                        st.write(f"  - {label} → {i}")
            
            # Unseen data results summary
            if st.session_state.unseen_results is not None:
                st.subheader("🔮 Unseen Data Summary")
                results = st.session_state.unseen_results
                st.write(f"**Total Predictions:** {len(results)}")
                
                pred_counts = results['Predicted_Label'].value_counts()
                st.write("**Prediction Counts:**")
                for label, count in pred_counts.items():
                    st.write(f"  - {label}: {count}")
    
    else:
        st.info("👆 Please upload a CSV file to get started")
        
        # Show instructions
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📋 How to Use This Dashboard")
            st.write("""
            1. **Upload Training Data**: Upload a CSV file with your dataset
            2. **Select Features & Target**: Choose input features and target variable
            3. **Configure Model**: Enable grid search for automatic hyperparameter tuning
            4. **Train Model**: Click 'Train SVM Model' to find the best parameters
            5. **Test Model**: Evaluate performance on the test set
            6. **Save Model**: Save the best model and parameters for later use
            7. **Test Unseen Data**: Upload new data to get predictions
            
            **Features:**
            - 🔍 Automatic hyperparameter tuning with Grid Search
            - 💾 Save and load trained models
            - 🔮 Test on completely unseen data
            - 📊 Detailed performance metrics and visualizations
            - 📥 Download prediction results
            """)
        
        with col2:
            st.subheader("🧠 Understanding SVM Parameters")
            
            with st.expander("🔧 Kernel Types", expanded=True):
                st.write("**Linear**: Straight decision boundaries")
                st.write("- Best for: Linearly separable data")
                st.write("- Example: Text classification, high-dimensional data")
                
                st.write("**RBF (Radial Basis Function)**: Circular boundaries")
                st.write("- Best for: Non-linear, complex patterns")
                st.write("- Example: Image recognition, complex datasets")
                
                st.write("**Polynomial**: Curved decision boundaries")
                st.write("- Best for: Moderately complex patterns")
                st.write("- Example: Biological data, engineering problems")
                
                st.write("**Sigmoid**: S-shaped boundaries")
                st.write("- Best for: Neural network-like problems")
                st.write("- Example: Binary classification tasks")
            
            with st.expander("⚖️ Key Parameters"):
                st.write("**C (Regularization)**")
                st.write("- Low C: Simpler model, may underfit")
                st.write("- High C: Complex model, may overfit")
                st.write("- Typical range: 0.1 to 100")
                
                st.write("**Gamma** (for RBF/Poly kernels)")
                st.write("- Low gamma: Smooth decision boundary")
                st.write("- High gamma: Complex decision boundary")
                st.write("- 'scale' or 'auto': Automatic calculation")
                
                st.write("**Degree** (for Polynomial kernel)")
                st.write("- Degree 2: Quadratic curves")
                st.write("- Degree 3: Cubic curves")
                st.write("- Higher degrees: More complex curves")
            
            with st.expander("💡 Tips for Better Results"):
                st.write("✅ **Use Grid Search** for automatic parameter tuning")
                st.write("✅ **Normalize your data** (done automatically)")
                st.write("✅ **Start with RBF kernel** for most problems")
                st.write("✅ **Use polynomial for moderate complexity**")
                st.write("✅ **Check validation accuracy** to avoid overfitting")

if __name__ == "__main__":
    main()