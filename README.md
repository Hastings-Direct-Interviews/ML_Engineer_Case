# ML_Engineer_Case

Hello! Thanks for your interest in ML Engineering at Hastings Direct. The below task is meant to give you an opportunity to demonstrate your experience in this area. 
So the below steps should be thought of as a purposefully loose "guide" that instructs the core of the case we wish for you to complete. But it does not limit you to implementing
or discussing other aspects that could be useful for a fully fledged ML Ops solution.

### Technical Task Context
•	Our data scientists have built a First Notification of Loss (FNOL) model that now needs to be deployed as an Azure ML service.  <br>
•	This model takes the information given at FNOL for a claiming policy holder and predicts what the ultimate cost will be for that claim. <br>
• Now the data scientist team has come to you knowing they have production code in this repo, but they need your help promoting into Azure services <br>
•	Your job is to use the files given here and create a CI/CD pipeline in Azure DevOps thats loads the model into a Docker container and deploys that container as an API accessible WebApp on Kubernetes


### Core Criteria for success
•	Candidate showcases an Azure DevOps pipeline
•	Candidate showcases the correct files of this repo being built into a Docker container
•	Candidate showcases the Docker container being managed by Kubernetes
•	Candidate showcases the API Endpoint being accesible and succesfully scoring an example record


### Tips
•	Please use Python 3.7 in your production environment as this is what the service has been tested with. <br>
•	Please start the task early as you will likely need to put in a request to Azure for free parallel jobs.  This request will take three working days to complete.


### Presentation Format
After completing the above task, compile your work into a short (15-25mins) presentation about the case. Be sure to demonstrate the core pieces discussed above.
But also consider what aspects of a full ML Ops solution that were not tested here and how you would approach those areas.
