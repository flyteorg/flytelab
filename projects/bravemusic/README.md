# Design Doc: Brave-Hyenas-2
## MLOps Community: Engineering labs

| Team name           |      brave-hyenas-2                      |
|---------------------|:----------------------------------------:|
|Project name         |      brave-hyenas-2                      | 
| Project description |     Hackathon - brave-hyenas-2 team      |
|Using GPUs? (Yes/No) |  No                                      | 
  
  
 
### Problem Statement
What problem are you solving?  
It’s usually hard to identify correctly what kind of music genre is playing thus our team embraced in tackling to classify music genre using deep learning.  
   

### Potential Solutions
At a high level. classifying music based on genres can help suggest the next songs to a listener, curate playlists of new recommendations, filter undesirable content or help streaming companies to control the music copyright for authors and many more. Our solution mainly focused on automating the training orchestration to attend these applicable solutions  


### Our Approach 
More and More music are been streamed and produced at high rate daily. Streaming companies like Spotify uses ML system to recommend songs and playlists to users but getting this system into the real world involves more than just building the models. in order to take full advantage of the built model we have to incorporate to focus on the core steps in building the model to make it effortless when the system is intact. Our Objective is to automate workflows using Flyte.  


### Dataset 
The Dataset we’re using is [GITZAN](http://marsyas.info/downloads/datasets.html) dataset which contains 1000 music files. It has ten types of genres with uniform distribution. Dataset has the following genres:  
1. blues
2. classical
3. country
4. disco
5. hiphop
6. jazz
7. reggae
8. rock
9. metal
10. pop.
Each music file is 30 seconds long. Which is hosted at http://opihi.cs.uvic.ca  

![Local dev drawio](https://user-images.githubusercontent.com/85021780/160123749-2cbeacb6-324e-42aa-8bb3-0bef06cc132c.png) 
  
    
![Option B drawio](https://user-images.githubusercontent.com/85021780/160124164-539dc8f9-c1e8-4b54-90e8-c761e3f0cb7d.png) 
   
#### solution
The following list highlights the steps of our level 0 process, as shown in Figure above:  
After long search & familiarization with flyte we decided to build a platform around flyte that could solve our use case. We came with these two options. Streamlit as a serving point after getting the modeled file vs passing the request to REST API i.e uvicorn to do the serving. For simplicity we choose Option A and left Option B for future work.  
1.	We load the data from our local storage to our workflow workspace
2.	With the GTZAN dataset loaded, We use Librosa library that is a helpful package to deal with audio files in Python as it transformations audio feature into MFCC.( The Mel-Scale defines frequency bands that are evenly distributed with respect to perceived frequencies. Mel filter banks are calculated so that they are more discriminative for lower frequencies) And For our Future work, the challenge would be to preprocess the audio data sufficiently enough to extract all essential features, even if that means applying several different techniques beyond MFCC. We train with 13 coefficients (n_mfcc=13) which is the simplest aspects of the spectral shape while using higher coefficients can be less important to our training. We only use (n_segments=10) just small portion to get each track to have (10 x 13 MFCC) equally distributed. This is all done in preprocess function that is under flyte @task which stores the associated labels in a separate json file (“data.json”) and to be called in the training function task

3.	The model is train under flyte task where the trained model file is stored as FlyteDirectory in temp location.

4.	The workflow argument is implicitly passed to run the decorated function to turn the task or construct the task as an output.
5.	 Manual, script-driven streamlit app access the model from the given model url after the workflow and output the flyte/temp location of the model.  
  

#### Challenges
Running Local is common in flyte workflows which are beginning steps to apply flyte workflows to any use cases. This manual, data-scientist-driven process might be sufficient when data are small or rarely changed or trained. In practice, workflow often break when they are deployed in the real world if not handled with care, To address these challenges and to maintain your workflow orchestration in production, we encountered with this challenges:  
1.	Flytekit task has parameters decorators, this stopped as to simply pass a function as a task 
2.	Couldn’t run workflow with non-task (flyte), Ideally it should ignore non-task function and just output the tasks while showing the DAGS for the non-task functions as outliners (just an opinion) 
3.	Working with Audio and preprocessing into mfcc vector json format while using the output of this task to be an input of another task was fairly timing consuming since at the beginning were wondering much about where the data we sit in the deployment cluster.
4.	The mindset of Kubernetes toughest (making abstractions around Kubernetes is touch i.e if things break, they break bad)
5.	We Struggled with communication and organize meet ups, This made as ended up with only two active members.
    
    This is How bad we planned:
  ![project scope drawio (1)](https://user-images.githubusercontent.com/85021780/160130318-9322fb98-a295-4731-a461-a240d2b21844.png)  
  
  One of the biggest challenges was working with 4-different time zones and ending up only two members to work on the project.   
  
### Production


![new](https://user-images.githubusercontent.com/85021780/161294904-a4158856-0558-424f-9f07-85aef8f4b423.jpg)  


The following figure depicts the proposed architecture. The goal is to build ML platform around flyte to perform continuous training of the model by automating the ML workflows; this lets you achieve continuous orchestration of model training. We use Github Registry to provide repo for our docker image and register to hosted cluster i.e Union.ai.  
 
### Solution (working progress)
1.	Script-driven, and interactive process: After testing our workflows locally we create github Package register to provide a central container registry.
    
       export CONTAINER_REPO_TOKEN="<your-token>"  
       echo $CONTAINER_REPO_TOKEN | docker login ghcr.io -u <your-username> --     password-stdin

2.	Flyte Uses this create container to package the workflows and tasks and send them to the remote Flyte cluster in our case union.ai. 
3.	Next packaging the workflow using deploy.py function to bundle with flyte backend. During this time the task is bound to an image that contain the code for the task & workflow which is registered with FlyteAdmin using registration task & workflow Api.
4.	The package file from the previous step is uploaded to Flyte backend. The referenced tasks in the packaged file are replaced by their FlyteAdmin registered identifier. All these associated entities are registered with the FlyteAdmin using the register workflow API. The ML engineer can then launch an excitation using the flyteadmin launch execution Api via Flyte Console or can be done using flytectl to launch, execute or monitor workflows 
5.	We planned to Create An interactive ui (streamlit) that is hosted in Heroku while getting access to Model file in Union.ai Cluster thus using secret management system of streamlit.  

       FLYTE_BACKEND = "remote"  # point the app to the playground backend  
       FLYTE_CREDENTIALS_CLIENT_ID = "<client_id>"  # replace this with your client id  
       FLYTE_CREDENTIALS_CLIENT_SECRET = "<client_secret>"  # replace this with your client secret
6.	Unfortunately Heroku has Slug Size after compression of 500 MB And our App exceeds assert Storage i.e S3. We should use cache to minimize this

### Future Work
•	CI/CD automation
•	Feature store
•	Model Monitoring