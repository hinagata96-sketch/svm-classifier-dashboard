# 🤖 SVM Classifier Dashboard

A comprehensive Streamlit web application for training, testing, and deploying Support Vector Machine (SVM) classifiers with automatic hyperparameter tuning and model management.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-v1.3+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Live Demo

**Try the live application:** [SVM Classifier Dashboard](https://svm-classifier-dashboard.streamlit.app)

> 🎯 **Quick Start:** Upload the included `sample_iris_data.csv`, select all features and the 'species' target, then click "Train SVM Model" to see it in action!

## 🚀 Features

- 📁 **Interactive File Upload**: Upload CSV datasets directly through the web interface
- 🔍 **Automatic Hyperparameter Tuning**: Grid Search CV to find optimal parameters
- 🎛️ **Manual Parameter Control**: Override automatic settings with custom parameters
- 🧠 **Multiple Kernels**: Support for linear, RBF, polynomial, and sigmoid kernels
- 📊 **Interactive Visualizations**: Decision boundary plots and confusion matrices
- 🎯 **Comprehensive Metrics**: Training, validation, and test accuracy with detailed reports
- 💾 **Model Persistence**: Save and load trained models for reuse
- 🔮 **Unseen Data Testing**: Apply trained models to completely new datasets
- 📥 **Export Results**: Download prediction results as CSV files
- 🧭 **Educational Content**: Detailed explanations of SVM parameters and concepts

## 📸 Screenshots

### Main Dashboard
![Dashboard](https://via.placeholder.com/800x400?text=SVM+Dashboard+Screenshot)

### Parameter Explanations
![Parameters](https://via.placeholder.com/600x300?text=Parameter+Explanations)

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/svm-classifier-dashboard.git
cd svm-classifier-dashboard
```

2. **Create and activate virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run svm.py
```

5. **Open your browser** and navigate to `http://localhost:8501`

## 🌐 Deployment

### Streamlit Cloud Deployment

This app is deployed on Streamlit Cloud and accessible at: [svm-classifier-dashboard.streamlit.app](https://svm-classifier-dashboard.streamlit.app)

#### Deploy Your Own Copy

1. **Fork this repository** on GitHub
2. **Go to [share.streamlit.io](https://share.streamlit.io)**
3. **Sign in** with your GitHub account
4. **Click "New app"**
5. **Select your forked repository**
6. **Set main file path:** `svm.py`
7. **Click "Deploy"**

### Alternative Deployment Options

#### Heroku
```bash
# Install Heroku CLI, then:
heroku create your-svm-dashboard
git push heroku main
```

#### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "svm.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📋 Usage

### Basic Workflow

1. **Upload Training Data**: Upload a CSV file with your dataset
2. **Select Features**: Choose input variables for training
3. **Select Target**: Choose the variable you want to predict
4. **Configure Model**: 
   - Enable Grid Search for automatic optimization
   - Or manually select kernel and parameters
5. **Train Model**: Click "🚀 Train SVM Model"
6. **Evaluate Performance**: Click "🧪 Test Model" for detailed metrics
7. **Save Model**: Click "💾 Save Best Model" for future use
8. **Test New Data**: Upload unseen data and get predictions

### Advanced Features

#### Grid Search vs Manual Selection
- **Grid Search**: Automatically tests multiple parameter combinations
- **Manual Selection**: Choose specific kernel, C, gamma, and degree values
- **Comparison**: Switch between modes to compare results

#### Model Management
- **Save Models**: Preserve trained models with timestamps
- **Load Models**: Restore previously saved models
- **Parameter History**: JSON files track all parameter combinations

#### Understanding Results
- **Decision Boundaries**: Visual representation of how the model separates classes
- **Confusion Matrix**: Detailed breakdown of prediction accuracy
- **Parameter Explanations**: Interactive guides explaining each SVM parameter

## 🧠 Understanding SVM Parameters

### Kernel Types

| Kernel | Description | Best For | Decision Boundary |
|--------|-------------|----------|-------------------|
| **Linear** | Straight lines | Linearly separable data, text classification | Straight |
| **RBF** | Radial basis function | Complex, non-linear patterns | Circular/curved |
| **Polynomial** | Polynomial functions | Moderately complex patterns | Polynomial curves |
| **Sigmoid** | Sigmoid function | Neural network-like problems | S-shaped |

### Key Parameters

- **C (Regularization)**: Controls trade-off between smooth boundary and correct classification
  - Low C (0.1): More tolerant of errors, smoother boundary
  - High C (100): Less tolerant of errors, complex boundary

- **Gamma**: Defines influence reach of training examples (RBF/Poly kernels)
  - Low gamma: Far-reaching influence, smoother boundaries
  - High gamma: Close influence, more complex boundaries

- **Degree**: Polynomial degree for polynomial kernel (2=quadratic, 3=cubic, etc.)

## 📊 Sample Data

The repository includes sample datasets for testing:

- `sample_iris_data.csv`: Classic Iris dataset for training
- `sample_unseen_data.csv`: New data for testing predictions

### Data Format Requirements

- CSV format with headers
- Numerical and categorical data supported
- Missing values should be handled before upload
- One-hot encoded targets are automatically detected

## 🔧 API Reference

### Command Line Usage

You can also use saved models programmatically:

```python
# Load and use a saved model
python use_saved_model.py

# Quick prediction
python quick_predict.py model_file.pkl data_file.csv
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Guidelines

- Follow PEP 8 style guidelines
- Add comments for complex logic
- Update documentation for new features
- Test thoroughly before submitting

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [Scikit-learn](https://scikit-learn.org/) for machine learning tools
- [Matplotlib](https://matplotlib.org/) for visualization capabilities

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/svm-classifier-dashboard/issues) page
2. Create a new issue with detailed description
3. Include error messages and steps to reproduce

## 🚀 Future Enhancements

- [ ] Support for more ML algorithms (Random Forest, Neural Networks)
- [ ] Advanced feature engineering tools
- [ ] Model comparison dashboard
- [ ] Batch prediction capabilities
- [ ] REST API for model serving
- [ ] Docker containerization

---

**Made with ❤️ and Python**
