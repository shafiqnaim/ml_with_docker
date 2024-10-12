# ml_with_docker

![image](https://github.com/user-attachments/assets/169b83b7-d051-4e76-92c0-9d4f9e58ea6c)
Step-by-Step Guide to Deploying ML Models
 

Let’s walk through how to deploy a machine-learning model using Docker.

 

1. Set Up Your Environment
Before you start, make sure you have installed Docker on your machine. You can download it from the Official Docker Website.

 

2. Build Your Machine Learning Model
You need to have a trained machine-learning model ready to be deployed. For this tutorial, we take a quick example in Python using scikit-learn.
```
model.py:

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Train and save the model
def train_model():
    # Load dataset
    data = load_iris()
    X, y = data.data, data.target
    
    # Train model
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # Save the trained model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Model trained and saved as model.pkl")

# Load model and make a prediction using predefined test data
def predict():
    # Load the saved model
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Test data (sample input for prediction)
    test_data = [5.1, 3.5, 1.4, 0.2]  # Example features
    prediction = model.predict([test_data])
    
    print(f"Prediction for {test_data}: {int(prediction[0])}")

if __name__ == '__main__':
    train_model()
    predict()
 
```
The above example combines model training, saving, and prediction in a single script. The train_model() function trains a simple model on the Iris dataset and saves it as “model.pkl”. While the predict() function loads the saved model and uses predefined test data to make predictions.

 

3. Create a requirements.txt File.
List down all the Python dependencies that your app requires in this file. In this case:
```
requirements.txt:
scikit-learn
``` 

4. Create a Dockerfile
The Dockerfile is a script that contains a series of instructions used to build a Docker image.
Following is the simple dockerfile for our app. Make sure that the dockerfile is created with no extension, as it allows Docker to recognize it without requiring any additional arguments when building an image.

```
Dockerfile:

# Use a base image with Python
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the necessary files into the container
COPY requirements.txt requirements.txt
COPY model.py model.py

# Install the required Python libraries
RUN pip install -r requirements.txt

# Run the Python script when the container starts
CMD ["python", "model.py"]
``` 

Now let’s understand what each of the keywords in the Dockerfile means.

- FROM: It specifies the base image for our Dockerfile.We are using Python 3.11-slim in our case.
- WORKDIR: It sets the working directory to the given path. After this, all commands will be executed relative to this directory.
- COPY: This command copies the contents from your local machine to the Docker container. Here, it’s copying requirements.txt and model.py files.
- RUN: It executes the command inside a shell (within the image's environment). Here, it is installing all the project dependencies listed in the requirements.txt file.
- CMD: This command specifies the default command to run when the container starts. It is running a model.py script using Python in this case.
 

5. Build a Docker Image
   Open your command prompt or terminal, navigate to the working directory where your Dockerfile is located, and run the following command.
```
docker build -t ml-model .
```

  This command builds a docker image named ml-model using the current directory.

 

6. Run the Docker Container
  Once the docker image is built, we are finally ready to run the container. Run the following command.
```
docker run ml-model
``` 

  Following is the output:
```
Model trained and saved as model.pkl
Prediction for [5.1, 3.5, 1.4, 0.2]: 0
``` 

7. Tag & Push the Container to DockerHub
  Docker Hub is a repository for Docker images, making it easy to share, version, and distribute containers across teams or production environments.

  Create an account on Docker Hub. Once you have it, log in through the terminal by running the following command.
```
docker login
``` 

  You have to tag the docker image with your username so that it will know where to push the image. Run the following command by replacing your username.
```
docker tag ml-model yourdockerhubusername/ml-model
``` 

Once the image has been tagged, you can push the image to the Docker hub by the following command.
```
docker push yourdockerhubusername/ml-model
``` 

Anyone can now pull and run your Docker image by:
```
docker pull yourdockerhubusername/ml-model
docker run yourdockerhubusername/ml-model
```

# [Reference](https://www.kdnuggets.com/step-by-step-guide-to-deploying-ml-models-with-docker)
