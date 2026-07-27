#Lab 4 : Implement a Keras Multi-Layer Perceptron (MLP) for Multiclass Classification

## 🎯 Aim

To implement a **Multi-Layer Perceptron (MLP)** using TensorFlow's Keras API for **multiclass classification** on the Iris dataset. This practical demonstrates how an Artificial Neural Network can classify input samples into one of multiple categories using hidden layers and the Softmax activation function.

---

## 📚 Main Topics Covered

- Artificial Neural Networks (ANN)
- Multi-Layer Perceptron (MLP)
- Multiclass Classification
- TensorFlow
- Keras Sequential API
- Dense Layers
- ReLU Activation Function
- Softmax Activation Function
- Sparse Categorical Crossentropy Loss
- Adam Optimizer
- Feature Scaling
- Model Evaluation

---

## 📖 Theory

A **Multi-Layer Perceptron (MLP)** is a type of Feedforward Artificial Neural Network consisting of an input layer, one or more hidden layers, and an output layer. Unlike a simple perceptron, an MLP can learn complex and non-linear relationships in data by using multiple hidden neurons and activation functions.

In this practical, the Iris dataset is used, which contains measurements of iris flowers belonging to three different species:

- Setosa
- Versicolor
- Virginica

Each flower is represented using four numerical features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The network learns patterns from these measurements and predicts the correct flower species.

The hidden layers use the **ReLU (Rectified Linear Unit)** activation function to introduce non-linearity, while the output layer uses the **Softmax** activation function to generate probability scores for all three classes.

The model is trained using the **Adam optimizer** and the **Sparse Categorical Crossentropy** loss function, which is well-suited for multiclass classification problems where class labels are represented as integers.

---

## ⚙️ Code Concept

The implementation follows these steps:

1. Import the required Python libraries.
2. Load the Iris dataset.
3. Split the dataset into training and testing sets.
4. Normalize the input features using StandardScaler.
5. Build the MLP model using the Keras Sequential API.
6. Add two hidden Dense layers with ReLU activation.
7. Add an output Dense layer with Softmax activation.
8. Compile the model using:
   - Adam Optimizer
   - Sparse Categorical Crossentropy Loss
9. Train the model for multiple epochs.
10. Evaluate the trained model on the test dataset.
11. Generate predictions.
12. Display:
    - Classification Report
    - Confusion Matrix
    - Accuracy Curve
    - Loss Curve

---

## 🖥️ Sample Output

```text
Dataset Shape: (150, 4)

Classes:
['setosa' 'versicolor' 'virginica']

Test Accuracy: 0.9667

Classification Report

              precision    recall    f1-score

setosa          1.00       1.00       1.00
versicolor      0.94       0.94       0.94
virginica       0.94       0.94       0.94

Overall Accuracy: 96.67%
```

The notebook also displays:

- Training Accuracy Curve
- Validation Accuracy Curve
- Training Loss Curve
- Validation Loss Curve
- Confusion Matrix

---

## 🔍 Observations

- The Multi-Layer Perceptron successfully classified the Iris flower species with high accuracy.
- Feature normalization improved training stability and convergence.
- The hidden layers enabled the network to learn complex relationships among the input features.
- The Softmax activation function produced probability values for each flower class.
- The Adam optimizer minimized the loss efficiently during training.
- The accuracy and loss graphs showed stable learning without significant overfitting.

---

## 🌍 Real-World Application

### Plant Species Classification

One practical application of Multi-Layer Perceptrons is **automatic plant species classification** in agriculture and environmental science.

Researchers often collect measurements such as:

- Leaf Length
- Leaf Width
- Petal Length
- Petal Width
- Stem Thickness

Instead of manually identifying plant species, these measurements can be supplied to a trained neural network.

The MLP analyzes the input features, identifies hidden patterns learned during training, and predicts the most probable plant species.

Such systems are useful for:

- Botanical Research
- Smart Agriculture
- Biodiversity Monitoring
- Forest Management
- Automated Plant Identification Apps

Although modern image-based plant classification commonly uses Convolutional Neural Networks (CNNs), Multi-Layer Perceptrons remain an excellent choice when structured numerical features are available.

---

## 🚀 Advantages

- Can classify multiple classes simultaneously.
- Learns complex, non-linear relationships.
- Easy to implement using TensorFlow and Keras.
- Efficient for structured numerical datasets.
- Forms the foundation for deeper neural network architectures.

---

## ⚠️ Limitations

- Performance depends on proper feature engineering and preprocessing.
- Sensitive to hyperparameter selection such as learning rate and number of hidden neurons.
- Less effective than CNNs for image-based tasks.
- Requires sufficient training data for good generalization.

---

## ✅ Conclusion

The Multi-Layer Perceptron (MLP) was successfully implemented using TensorFlow and Keras for multiclass classification on the Iris dataset. The model effectively learned patterns from the input features and accurately classified flower species using the Softmax activation function. This practical demonstrated the complete workflow of a multiclass neural network, including preprocessing, model construction, training, evaluation, and visualization of results. It provides a strong foundation for solving more advanced classification problems using deep learning.

---

## 🛠️ Technologies Used

- Python
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow
- Keras
- Jupyter Notebook

---

## 🎓 Key Learning Outcomes

After completing this practical, the following concepts were understood:

- Multi-Layer Perceptron Architecture
- Multiclass Classification
- Feature Scaling using StandardScaler
- Dense Layers
- ReLU Activation Function
- Softmax Activation Function
- Sparse Categorical Crossentropy Loss
- Adam Optimizer
- Model Training and Validation
- Confusion Matrix
- Classification Report
- Accuracy and Loss Visualization
