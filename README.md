# mlops

This is one of the approaches to build / tune your own model and deploy.
Pls refer MLOPs Flow.xlsx first for an easier understanding of the MLOps workflow.
All related code is available in the "code" folder.
The "2. Steps to trigger the mlops-flow State Machine.txt" is the first step in the workflow that triggers the State Machine.

Note:
If metrics are below a certain thereshold, the endpoint will be automatically deleted. 
So, the evaluation metrics must be above a defined threshold for the endpoint to be available for usage.
