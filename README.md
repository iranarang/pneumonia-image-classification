# Pneumonia Chest-X-Ray Classification

This repository packages a highly accurate Convolutional Neural Network (CNN) that detects radiological signs of pneumonia in pediatric chest-X-rays. The model was automatically tuned with Keras Tuner and found that Random Search was the best performing tuning method. A Flask API containarized with Docker exposes the model for easy, reproducible inference. 

## Repository Contents

1.**`pneumonia_classification.ipynb`**
   This notebook loads the dataset, performs EDA, builds a baseline CNN, runs Keras Tuner (Random Search, Hyperband, Bayesian), and retrains the best configuration and saves the final model. 
   
2. **`api.py`**
  
   This file defines a Flask application with three routes:
   - **`/summary`**: Returns metadata about the model using a **GET** request.
   - **`/best-hyperparameters`**: Returns optimal hyperparameters discovered by KerasTuner using a **GET** request
   - **`/inference`**: Accepts an image file and returns "normal" or "pneumonia" using a **POST** request.

3. **`Dockerfile`**

   Creates a Docker image, installs dependencies, and copies the application files.

4. **`docker-compose.yml`**

   Simplifies use of the Docker container.

5. **`dataset/`**

   Contains sample damaged and not damaged satellite images.   

# Data

We use the **Chest X-Ray Images (Pneumonia)** dataset (Kermany *et al.*, 2018) comprising **5 863 anterior–posterior paediatric X-rays** split as follows:

| Split       | Images |
|-------------|--------|
| Train       | 5 216  |
| Validation  | 16     |
| Test        | 624    |

All images are resized to **150 × 150 × 3** and rescaled to **[0, 1]** before being fed into the network.

## Model Architecture

The Keras Tuner search produced the following winning configuration:

| Hyperparameter       | Value              |
|----------------------|--------------------|
| Convolutional blocks | 3                  |
| Filters              | 64 → 128 → 128     |
| Dense units          | 256                |
| Drop-out             | 0.4                |
| Optimizer / LR       | Adam @ 0.01        |

With these settings the network scored **100 % recall** on the validation set – a key requirement in medical screening tasks where false negatives are unacceptable.


## Building Instructions

Before building, ensure **Docker** and **Docker Compose** are installed on your machine. You can check this by typing, 
```
docker --version
docker-compose --version
```

Then, clone this repo by writing the following in terminal,
```
git clone git@github.com:iranarang/pneumonia-image-classification.git
```
Now, in order to build the Docker images and start the container, 
```
docker compose -f docker/docker-compose.yml up --build -d
```

Now, the server is up and running! We can test our requests using the following curl commands. The first curl command is our GET request.

```
curl http://localhost:5000/summary
```

This will return,
```
{
  "accuracy": 1.0,
  "description": "A CNN that classifies chest X-ray images as normal vs. pneumonia",
  "name": "pneumonia-detection-cnn",
  "version": "v1"
}
```
Now, we can test our next GET request using the command below.

```
curl http://localhost:5000/best-hyperparameters
```
This will return,
```
{
  "Conv Blocks": 3,
  "Filters 0": 64,
  "Filters 1": 128,
  "Filters 2": 128,
  "Dense Units": 256,
  "Dropout": 0.4,
  "Optimizer": "adam"
  "Learning rate": 0.01
}
```

Now, we can test our POST request. The command below is the format of our curl command.
```
curl -X POST -F "image=@example_directory/example_file.jpeg" http://localhost:5000/inference
```
An actual example of this is seen as,
```
curl -X POST -F "image=@/home/ubuntu/pne/data/val/pneumonia/person1946_bacteria_4874.jpeg" http://localhost:5000/inference
```
This outputs,
```
{
  "prediction": "pneumonia"
}
```
Once we are done running the server, we can close it by typing the following in terminal
```
docker compose -f docker/docker-compose.yml down
```

## References
Mooney, Paul. “Chest X-Ray Images (Pneumonia).” Kaggle, 24 Mar. 2018, www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download. 
