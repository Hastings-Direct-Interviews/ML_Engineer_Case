# ML_Engineer_Case

 

Hello! Thanks for your interest in ML Engineering at Hastings Direct. The below task is meant to give you an opportunity to demonstrate your experience in this area. 
So the below steps should be thought of as a purposefully loose "guide" that instructs the core of the case we wish for you to complete. But it does not limit you to implementing
or discussing other aspects that could be useful for a fully fledged ML Ops solution.

 

### Technical Task Context

•       Our data scientists have built a model which predicts the final cost of claims based on the information available when a customer first reports their claim to Hastings Direct.  This model now needs to be deployed as an Azure ML service.  <br>
•       By predicting what the ultimate cost will be for a claim, we can ensure that we set aside enough money (known as claim reserves) to ensure we fulfil our promise to the customer such as repairing their vehicle and/or compensating them for any injuries they may have recieved <br>
• Now the data science team has come to you with what should be production ready code in this repo, but they need your help promoting into Azure services <br>
•       Your job is to use the files given here and create a CI/CD pipeline in Azure DevOps thats loads the model into a Docker container and deploys that container as an API accessible WebApp on Kubernetes

 

 

### Core Criteria for success

•       Candidate showcases an Azure DevOps pipeline <br>
•       Candidate showcases the correct files of this repo being built into a Docker container via the DevOps pipeline <br>
•       Candidate showcases the Docker container being managed by Kubernetes <br>
•       Candidate showcases the API Endpoint being accessible and succesfully scoring an example record <br>
•       Candidate demonstrates considerations for scalability, availability, disaster recovery, costs, security & monitoring <br>

 

### Tips

•       Please use Python 3.7 in your production environment as this is what the service has been tested with. <br>
•       Please start the task early as you will likely need to put in a request to Azure for free parallel jobs.  This request will take three working days to complete. <br>
• While preferred, Azure services are not required for this task. If you are more familiar with a competing cloud service provider, completing this task in that stack is acceptable. <br>
• While not required, we prefer infrastructure as code solutions, so that pipelines can be configured and re-used. <br>
• Please feel free to reach out to your recruiter with questions about the case. The team behind this case will decide if it's "giving away too much" or fair to answer.


### Presentation Format

After completing the above task, compile your work into a short (15-25mins) presentation which demonstrates your thought process. Be sure to demonstrate the core criteria discussed above.
But also consider what aspects of a full ML Ops solution that were not tested here and how you would approach those areas. At a minimum please discuss what went well, what didn't go well, how the production code could potentially be improved, and how you would ensure model performance would be maintained over time.
