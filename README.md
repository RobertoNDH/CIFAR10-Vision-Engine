# CIFAR10-Vision-Engine

A professional image recognition engine built with Streamlit and TensorFlow 2. This application allows users to classify images into CIFAR-10 categories using a Convolutional Neural Network (CNN).

## Features
- Responsive split-column interface.
- Interactive bar charts generated with Plotly to display the top 3 predictions.
- Modular architecture separating inference logic (`model_engine.py`) from the UI (`iaAPP.py`).
- Error handling: Image format validation (auto-conversion to RGB) and exception handling for model loading.
- Configured for deployment with `requirements.txt` and Streamlit Cloud configuration (`config.toml`).

## Project Structure

```text
├── iaAPP.py              
├── model_engine.py       
├── modelo_1.py           
├── requirements.txt      
├── .streamlit/           
│   └── config.toml       
└── modelo_cifar10_final.keras 
```

## Installation

### 1. Clone or download the repository
Navigate to the project folder:
```bash
cd path/to/your/project
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
Ensure pip is updated and install the required packages:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

With the virtual environment activated, run the following command:

```bash
streamlit run iaAPP.py
```

The browser will automatically open at `http://localhost:8501`. 

### Steps:
1. Review the left sidebar for the available classification categories.
2. Drag and drop or browse for an image file (.jpg, .jpeg, or .png).
3. The image will be processed automatically.
4. Review the top 3 predictions and probabilities on the right panel.

## Simulated Interface

> **Main View**

```text
+-------------------------------------------------------------+
| Model Details                     | Image Recognition       |
| [TF Logo]                         |                         |
|                                   | Upload an image for     |
| Model: CNN                        | analysis.               |
| Dataset: CIFAR-10                 |                         |
|                                   | +---------------------+ |
| Classes it can recognize:         | | Choose an image file| |
| - Airplane                        | |    Browse files     | |
| - Automobile                      | +---------------------+ |
| - Bird                            |                         |
| - Cat                             |                         |
...
```

> **Prediction View**

```text
+-------------------------------------------------------------+
| Model Details           | Uploaded Image    | Prediction Analysis|
|                         |  +-----------+    |                    |
| Model: CNN              |  |           |    | Top Match: Dog     |
| Dataset: CIFAR-10       |  | [Dog.jpg] |    |                    |
|                         |  |           |    |                    |
| Classes it can recognize|  +-----------+    | Dog  ███████ 94%   |
| - Airplane              | Format: JPEG      | Cat  ██▏     18%   |
| - Automobile            |                   | Deer ▏        4%   |
...
```

## Training

To retrain the model or modify the architecture, execute:

```bash
python modelo_1.py
```
This script uses Keras native preprocessing layers and tf.data, and includes EarlyStopping. The best configuration will be saved as `mejor_modelo_cifar10.keras`.
