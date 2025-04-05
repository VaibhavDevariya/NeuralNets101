# PyTorch Implementations in NeuralNets101

Welcome to the **PyTorch** folder of the **NeuralNets101** repository! This folder contains various neural network implementations using **PyTorch**. Below, you'll find descriptions of the neural networks included in this folder:

## Neural Networks Implemented

1. **Convolutional Neural Network (CNN) for Fashion MNIST** : This model implements a Convolutional Neural Network (CNN) for image classification using the Fashion MNIST dataset.
- **Notebook**: [fmnist_CNN.ipynb](fmnist_CNN.ipynb)

2. **Feedforward Neural Network (NN) for Fashion MNIST** : This is a simple Feedforward Neural Network (NN) that classifies images from the Fashion MNIST dataset without using convolutional layers. The model relies purely on fully connected layers (also known as dense layers) for learning representations from the images.
- **Notebook**: [fmnist_NN.ipynb](fmnist_NN.ipynb)

3. **Summation Model (NN)** : This is a basic Feedforward Neural Network (NN) that learns to sum two numbers. The network takes two numeric inputs and learns to predict their sum. It demonstrates how a simple neural network can solve a basic arithmetic problem, showcasing how neural networks can model mathematical operations.
- **Notebook**: [addition_NN.ipynb](addition_NN.ipynb)
---

## How to Use the PyTorch Notebooks

To get started with the PyTorch notebooks, follow these steps:

1. **Set Up Your Environment**: Ensure you've followed the instructions in the main `README.md` to set up your Python environment and install dependencies.
2. **Explore the Notebooks**: 
   - Navigate to the `pytorch` folder and open the desired Jupyter notebook.
   - Run the cells in the notebook to train the models on your local machine.
3. **Experiment**: Feel free to modify the code, experiment with different architectures, or adjust hyperparameters to see how the models perform with different configurations.

---

## Additional Notes

- **Model Performance**: The models in this repo serve as simple demonstrations of PyTorch's capabilities. Do not consider the Accuracy as the final accuracy.
- **Next Steps**: After experimenting with these networks, consider exploring more advanced architectures (like RNNs or GANs) or using different datasets to further deepen your understanding of PyTorch.