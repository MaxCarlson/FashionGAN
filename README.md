# FashionGAN

This project implements a Generative Adversarial Network (GAN) to generate images of fashion items. The GAN is trained on fashion datasets to learn the underlying distribution and generate realistic images of clothing.

## Project Overview

- **Generative Adversarial Network (GAN)**: This project utilizes a GAN architecture implemented with TensorFlow and Keras to generate images of fashion items.
- **Fashion Dataset**: The project uses fashion-related datasets for training the GAN to generate realistic images.
- **Visualization**: After training, the project generates and visualizes sample images of fashion items.

## Project Structure

- **FashionGAN.py**: The main Python script that defines and trains the GAN model on the fashion dataset.
- **e_10.jpg**: A sample generated image from the GAN after 10 epochs of training.

## Installation

### Prerequisites

- **Python 3.x**: Ensure Python 3.x is installed on your machine.
- **Required Libraries**: Install the required libraries listed in `requirements.txt`.

### Setup Instructions

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/YourUsername/FashionGAN.git
    cd FashionGAN
    ```

2. **Install Dependencies**:
    Install the necessary dependencies for running the project:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

1. **Train the GAN**:
    Execute the GAN training script:
    ```bash
    python FashionGAN.py
    ```

2. **Generate Images**:
    The script will generate images of fashion items after training the GAN. Sample images will be saved in the project directory.

## Project Workflow

1. **Data Preparation**: Load and preprocess the fashion dataset.
2. **Model Training**: Train the GAN using the fashion dataset to generate realistic fashion images.
3. **Image Generation**: Visualize and save generated images after each epoch of training.
