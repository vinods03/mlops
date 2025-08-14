import sys
import boto3
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
import time

s3_client = boto3.client('s3')
sagemaker_client = boto3.client('sagemaker')

bucket_name = 'data-us-west-2-100163808729'
predictions_path = 'system-test-output/predictions.csv'
actuals_path = 'system-test-output/actuals.csv'

endpoint_name = 'diamond-price-predictor-endpoint'
endpoint_config_name = 'diamond-price-predictor-endpoint-config'
model_name = 'diamond-price-predictor-model'

########################################### PREDICTIONS FROM ENDPOINT ###################################################

predictions_s3_response = s3_client.get_object(Bucket = bucket_name, Key = predictions_path)
predictions = predictions_s3_response['Body'].read().decode('utf-8')

# print(predictions)
# predictions_df = pd.DataFrame(list(map(lambda x: x.split(','), predictions.split('\n'))))
# predictions_df = predictions_df.iloc[:, 0]

predictions_df = pd.DataFrame(list(map(lambda x: x, predictions.split('\n'))))
predictions_df.replace('', np.nan, inplace=True)
predictions_df.dropna(inplace=True)

print(predictions_df.head())
print(predictions_df.count())

########################################### ACTUAL TEST DATA ###################################################

actuals_s3_response = s3_client.get_object(Bucket = bucket_name, Key = actuals_path)
actuals = actuals_s3_response['Body'].read().decode('utf-8').strip() # there was an extra row in actuals_df. this strip, along with drop tail(1) below, removed the last empty row

print(actuals)
# actuals_df = pd.DataFrame(list(map(lambda x: x.split(','), actuals.split('\n'))))
# actuals_df = actuals_df.iloc[:, 0]

actuals_df = pd.DataFrame(list(map(lambda x: x, actuals.split('\n'))))
actuals_df.drop(actuals_df.tail(1).index,inplace = True) # option 1 tried to remove last empty row
print(actuals_df.tail())
print(actuals_df.count())

mae = metrics.mean_absolute_error(actuals_df, predictions_df)
mse = metrics.mean_squared_error(actuals_df, predictions_df)
rmse = metrics.mean_squared_error(actuals_df, predictions_df, squared = False)
r_squared = metrics.r2_score(actuals_df, predictions_df)
# adjusted_r_squared=1-(((1-r_squared)*(53940-1))/(53940-26-1))

print('mae: ' +str(mae))
print('mse: ' +str(mse))
print('rmse: ' +str(rmse))
print('r_squared:' +str(r_squared))
# print('adjusted_r_squared: ' +str(adjusted_r_squared))

if (r_squared < 0.99):
    # Delete endpoint - this approach resulted in exception even when delete endpoint was successful. better approach below -> while condition is different
    # try:
    #     delete_endpoint_response = sagemaker_client.delete_endpoint(EndpointName = endpoint_name)
    #     while delete_endpoint_response['EndpointStatus'] == 'Deleting': # this statement has potential to result in exception at the point the endpoint is deleted
    #         print('Delete endpoint status is: ', delete_endpoint_response['EndpointStatus'])
    #         time.sleep(30)
    #     print('Delete endpoint completed')
    #     print('Delete endpoint response is: ', delete_endpoint_response)
    # except Exception as e:
    #     print('Delete endpoint response is: ', delete_endpoint_response)
    #     print("Delete endpoint failed with exception ", e)
        
    try:
        delete_endpoint_response = sagemaker_client.delete_endpoint(EndpointName = endpoint_name)
        while 'EndpointStatus' in delete_endpoint_response:
            print('Delete endpoint status is: ', delete_endpoint_response['EndpointStatus'])
            time.sleep(30)
        print('Delete endpoint completed')
    except Exception as e:
        print('Delete endpoint response is: ', delete_endpoint_response)
        print("Delete endpoint failed with exception ", e)
        
    # Delete endpoint config
    try:
        delete_endpoint_config_response = sagemaker_client.delete_endpoint_config(EndpointConfigName = endpoint_config_name)
        print('Delete endpoint config response is: ', delete_endpoint_config_response)
    except Exception as e:
        print("Delete endpoint config failed with exception ", e)
        
    # Delete model
    try:
        delete_model_response = sagemaker_client.delete_model(ModelName = model_name)
        print('Delete model response is: ', delete_model_response)
    except Exception as e:
        print("Delete model failed with exception ", e)
    