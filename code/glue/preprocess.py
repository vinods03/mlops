import sys
import os
import boto3
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# s3 = boto3.client('s3')
# object_key = 'input/diamonds.csv'
# input_object = s3.get_object(Bucket = bucket_name, Key = object_key)
# input_body = input_object['Body']
bucket_name = 'data-us-west-2-100163808729'
prefix = 'feature-engg-output'

# read the input data
input_data = pd.read_csv('s3://data-us-west-2-100163808729/input/diamonds.csv')
# input_data = input_data.reset_index(drop = True, inplace = True) -> If you do this, input_date is not a dataframe anymore
# input_data = input_data.iloc[:, 1:] -> This is dropping the price column instead of the index column
print('Check input data')
print(input_data.count())
print(input_data.head())

# encoding of categorical variables
input_data = pd.get_dummies(input_data)
print('encoding of categorical variables')
print(input_data.head())

train_df, validation_df, test_df = np.split(input_data.sample(frac=1, random_state=1729), [int(0.7 * len(input_data)), int(0.9 * len(input_data))])   

train_df.to_csv('train.csv', index = False, header = False)
validation_df.to_csv('validation.csv', index = False, header = False)
test_df.to_csv('test.csv', index = False, header = False)

print('Test features of one record')
test_features_single_record_df = test_df.head(1)
test_features_single_record_df.drop(['price'], axis = 1, inplace = True)
# print(type(test_features_single_record_df))
# print(test_features_single_record_df)
test_features_single_record_df.to_csv('test_features_single_record.csv', index=False, header=False)

print('Test features of all records')
test_features_all_records_df = test_df.drop(['price'], axis = 1)
# print(type(test_features_all_records_df))
# print(test_features_all_records_df)
test_features_all_records_df.to_csv('test_features_all_records.csv', index=False, header=False)

boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'validation/validation.csv')).upload_file('validation.csv')
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'test/test.csv')).upload_file('test.csv')

boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'test/test_features_single_record.csv')).upload_file('test_features_single_record.csv')
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'test/test_features_all_records.csv')).upload_file('test_features_all_records.csv')
