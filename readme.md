
# Jagriti via Mudra: A Pose-Based Surveillance Anomaly Detection System

## Overview

Jagriti via Mudra is a real-time anomaly detection system designed for video processing, focusing on human pose analysis. It addresses the limitations of existing methodologies, such as Generative Adversarial Networks (GANs), which fail to differentiate anomalies based on human pose, leading to inaccuracies in detecting complex activities. Our system integrates custom and publicly available datasets to train models for pose estimation and anomaly detection, ensuring robust and accurate detection of anomalies in real-time.

## Key Features

- **Custom and Public Datasets Integration**: Utilizes a mix of custom datasets tailored to specific research objectives and publicly available datasets like KTH, UCF, IXMAS, and JHMDB for comprehensive training.
- **Real-Time Anomaly Detection**: Processes video data in real-time, analyzing sequences of 10 frames at a time for efficient anomaly detection.
- **Pose Estimation and Action Recognition**: Tracks individuals across different frames or segments of video, identifying patterns of behavior or detecting anomalies specific to certain individuals.
- **Advanced Analytics**: Employs LSTM networks for pose-based details extraction and CNNs for RGB analysis, enabling precise monitoring of movements and accurate identification of objects.
- **Fully Connected Deep Neural Network (FC-DNN)**: Merges feature representations from both the POSE stream and RGB data stream using an FC-DNN, capturing a complete understanding of the scene for robust anomaly detection.

## Performance Metrics

The system's performance is evaluated using precision, recall, and F1-score, ensuring high accuracy in detecting anomalies. The confusion matrix provides a comprehensive overview of the classifier's effectiveness, including true positives, false positives, true negatives, and false negatives.

## Training and Validation

The system's training and validation loss and accuracy are crucial metrics, indicating the model's learning efficiency and its performance on unseen data. Balancing these metrics helps in creating models that generalize well to new data.

## Conclusion and Future Scope

Jagriti via Mudra has demonstrated significant potential in the field of video processing and anomaly detection. Future research directions include exploring the impact of different people on the results, integrating real-time anomaly detection with wearable devices, and exploring its applications in various domains such as security, healthcare, and entertainment.

## Getting Started

To get started with Jagriti via Mudra, follow the installation and setup instructions provided in the repository. Contributions and feedback are welcome!
### Installation

To set up the environment for Jagriti via Mudra, follow these steps:

1. Create a new conda environment named `jagrithiVM` with Python 3.8:
   ```
   conda create --name jagrithiVM python=3.8 -y
   ```
2. Activate the newly created environment:
   ```
   conda activate jagrithiVM
   ```
3. Install CUDA Toolkit 11.8:
   ```
   conda install cudatoolkit=11.8 -y
   ```
4. Install TensorFlow 2.10:
   ```
   pip install "tensorflow==2.10"
   ```
5. Install PyTorch, torchvision, torchaudio, and PyTorch CUDA 11.8:
   ```
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```
6. Install additional Python packages:
   ```
   pip install seaborn opencv-contrib-python scikit-learn ultralytics chardet
   ```
7. Verify the installation by checking the availability of GPU devices for TensorFlow and PyTorch:
   ```
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'));import torch; print(torch.cuda.is_available())"
   ```


## Contributing

We welcome contributions from the community. Please read our contributing guidelines for more information.

