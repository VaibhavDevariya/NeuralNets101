# NeuralNets101

Welcome to **NeuralNets101** – a repository designed to help beginners and enthusiasts learn and understand the basic forms of various neural networks! This repo offers implementations of fundamental neural networks using both **PyTorch** and **Keras**. Whether you're just starting with machine learning or looking to solidify your understanding, **NeuralNets101** provides a hands-on approach to understanding the core concepts behind neural networks.

## Key Features
- **Neural Network Implementations**: Basic implementations of popular neural networks like Feedforward Networks, Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), and more.
- **Dual Framework Support**: Each neural network is implemented in **PyTorch** and **Keras** (TensorFlow), so you can learn how to build and train the same neural net using both frameworks.
- **Beginner Friendly**: Code is simple and well-documented, making it easy to follow and understand the inner workings of each neural network.
- **Future Expansions**: As the repository grows, expect more neural network architectures, additional machine learning techniques, and exploration of other deep learning frameworks.

## Why NeuralNets101?

Neural networks can seem complex at first, but by breaking them down into their basic components and learning how to implement them from scratch, you gain a deeper understanding of how they work. This repo helps you get comfortable with the building blocks of deep learning and provides clear examples using two of the most popular deep learning frameworks: **PyTorch** and **Keras**.

### What You’ll Learn
- How neural networks are structured and work from scratch.
- The differences between PyTorch and Keras implementations.
- How to train and evaluate models in both frameworks.
- Key deep learning concepts like backpropagation, activation functions, optimization techniques, and more.

## How to Use This Repository

To get started with this repository, you'll need to set up your development environment. You can do this using either **Anaconda** or **Python** (using `venv`).

### 1. **Install Anaconda or Python**
- **Installing Anaconda** :  You can download Anaconda from [here](https://www.anaconda.com/products/individual).
- **Installing Python** : You can download Python from [here](https://www.python.org/downloads/).

---

### 2. **Create a Virtual Environment with dependencies**
Depending on whether you have **Anaconda** or **Python** installed, follow one of the steps below:

#### **If you have Anaconda installed**:
- Create a virtual environment using the `environment.yml` file provided in the repository.
    ```bash
    conda env create -f environment.yml
    ```
- Once the environment is created, activate it:
    ```bash
    conda activate neuralnets101
    ```

#### **If you only have Python installed**:
- Create a virtual environment using Python's `venv`:
    ```bash
    python -m venv neuralnets101
    ```
- Activate the virtual environment:
    - On **Windows**:
        ```bash
        .\neuralnets101\Scripts\activate
        ```
    - On **macOS/Linux**:
        ```bash
        source neuralnets101/bin/activate
        ```
- Install dependencies
    ```
    pip install -r requirements.txt
    ```
---

### 3. **Explore the Notebooks**
- Inside the repository, you will find two main folders: **`pytorch`** and **`keras`**.
- Each folder contains Jupyter Notebooks for different neural network implementations (e.g., Feedforward Neural Networks, Convolutional Neural Networks etc.) for various projects or datasets.
- The notebook files are named in the format:  
  `<projectname_or_datasetname>_<NN_or_CNN_or_GAN>.ipynb`
  
For example:
  - **pytorch/`mnist_CNN.ipynb`**: A Convolutional Neural Network implementation for the MNIST dataset using PyTorch.
  - **keras/`fmnist_CNN.ipynb`**: A Convolutional Neural Network implementation for the Fashion MNIST dataset using Keras.

You can open the respective notebook based on the framework you wish to work with and start experimenting with the code and models.

## Neural Networks Covered
- **Neural Network (NN)**: Simple nerual net with only neuron layers and activation function.
- **Convolutional Neural Networks (CNN)**: Understand how CNNs work for image classification tasks.
- **And more**: Future additions will include more advanced networks like RNNs, GANs, Autoencoders, and more.

## Future Additions

I am continually expanding the repository to include:
- More neural network architectures.
- More frameworks (e.g., TensorFlow, MXNet).
- Additional content to help you understand various aspects of machine learning and deep learning, from data preprocessing to model evaluation.

## Contributing

I welcome contributions! If you’d like to add a new neural network implementation or have suggestions for improvements, feel free to open an issue or submit a pull request.

- Fork the repo.
- Create a new branch (`git checkout -b feature-name`).
- Implement your changes.
- Push to your fork (`git push origin feature-name`).
- Create a pull request.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

Thank you for visiting **NeuralNets101**! We hope this repository helps you on your journey to mastering neural networks and deep learning. Happy coding! 🚀
