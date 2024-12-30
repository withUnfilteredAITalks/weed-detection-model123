# Weed Detection and Removal Using CNN

## Overview
This project aims to develop a machine learning model that can automatically identify weeds in agricultural fields. The model uses a Convolutional Neural Network (CNN) for image classification, distinguishing between crops and weeds in images of agricultural fields. The goal is to provide an automated system that helps farmers reduce pesticide use and manage weeds efficiently, ultimately improving crop yields.



## Group Members
- **Aman Reshid     Ugr/22667/13**
- **Amanuel Mergia  UGR/22530/13**
- **Ebisa Bette     Ugr/22643/13**
- **Desalegn Sisay  Ugr/23232/13**



## Requirements
The following libraries are required to run this project:

- `tensorflow`: Deep learning framework used for building and training the CNN model.
- `numpy`: For handling arrays and mathematical operations on images.
- `opencv-python`: For image processing tasks such as resizing, normalizing, and loading images.
- `matplotlib`: Used for visualizing training progress (e.g., loss and accuracy graphs).
- `scikit-learn`: For machine learning utilities such as label encoding.



## Installation Instructions
To set up this project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/weed-detection.git
    cd weed-detection
    ```

2. **Create and activate a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use 'venv\Scripts\activate'
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

Ensure you have Python 3.6+ installed. If you are using a virtual environment, make sure it is activated before running the above commands.



## Dataset
The dataset contains images of weeds and crops, which are used to train the model for classification. 


Each subfolder should contain images for that respective class. Optionally, you can also create a separate `test_images/` folder for evaluating the model after training.




### Dataset Source
You can find the dataset [https://www.kaggle.com/datasets/jaidalmotra/weed-detection/code]


## Model Architecture
The model is built using a Convolutional Neural Network (CNN) with the following architecture:

- **Input Layer**: The images are resized to 224x224x3 pixels for input.
- **Conv2D Layers**: Convolutional layers that extract features like edges and textures from the images.
- **MaxPooling Layers**: These layers reduce the dimensionality of the feature maps while preserving essential information.
- **Flatten Layer**: Converts the feature maps into a 1D vector.
- **Dense Layers**: Fully connected layers to make classification decisions.
- **Dropout Layer**: Prevents overfitting by randomly setting some of the input units to 0 during training.
- **Softmax Output Layer**: Converts the model's output into probabilities for multi-class classification (weed or crop).



### Hyperparameters:
- **Input Size**: 224x224 pixels
- **Optimizer**: Adam
- **Loss Function**: Sparse categorical cross-entropy
- **Metrics**: Accuracy



## Training the Model
To train the model, follow these steps:

1. **Prepare the dataset**: Organize the images into the `train/weed/` and `train/crop/` directories.
2. **Run the training script**:
    ```bash
    python train_model.py
    ```
3. **Monitor the training**: The script will print the training and validation accuracy and loss over the epochs. You can also visualize these metrics using `matplotlib`.

Once training is complete, the model will be saved in the `saved_model/` directory.



## Testing the Model
After training the model, you can test it on new images to evaluate its performance. To do so:

1. Place your test images in the `test_images/` directory.
2. Run the testing script:
    ```bash
    python test_model.py --images_path test_images/
    ```

This will output predictions indicating whether the image contains a weed or a crop.



## Model Evaluation
The model’s performance can be evaluated using various metrics such as:

- **Accuracy**: The proportion of correctly classified images.
- **Confusion Matrix**: A matrix showing the true positives, false positives, true negatives, and false negatives.
- **Loss and Accuracy Graphs**: Visualize the model's training progress by plotting the loss and accuracy metrics.

For evaluating the model, you can use the following command after training:
```bash
python evaluate_model.py
