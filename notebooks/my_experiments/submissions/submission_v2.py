
# python submission.py --username gurvsin3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
from os import environ
import pwd

import pandas as pd
import numpy as np
import random
import re

import json
import os
import getpass

from datetime import datetime
import dsx_core_utils, requests, jaydebeapi, os, io, sys
from pyspark.sql import SparkSession
import pandas as pd

RESULT = {}

FLAGS = None

DATA_DIR = os.environ['DSX_PROJECT_DIR']+'/datasets'
# DATA_DIR = 'data'

def init_steps():
    global dataSet
    global dataSource
    global conn
    dataSet = dsx_core_utils.get_remote_data_set_info('submissions')
    dataSource = dsx_core_utils.get_data_source_info(dataSet['datasource'])
    if (sys.version_info >= (3, 0)):
    	conn = jaydebeapi.connect(dataSource['driver_class'], dataSource['URL'], [dataSource['user'], dataSource['password']])
    else:
    	conn = jaydebeapi.connect(dataSource['driver_class'], [dataSource['URL'], dataSource['user'], dataSource['password']])
    load_dataset()

def load_dataset():
  global submissions_df
  query = 'SELECT * FROM ' + (dataSet['schema'] + '.' if (len(dataSet['schema'].strip()) != 0) else '') + dataSet['table']
  submissions_df = pd.read_sql(query, con=conn)
  print(submissions_df.head())

def save_record(dataframe):
  print("IN save_record >>>>>>>>")
  for i in range(0, len(dataframe)):
      if dataframe["USERNAME"][i] == FLAGS.username:
          insert_query = "INSERT INTO "+ (dataSet['schema'] + '.' if (len(dataSet['schema'].strip()) != 0) else '') + dataSet['table']
          insert_query = insert_query + "(RANK, USERNAME, ACCURACY, ENTRIES, LAST) VALUES ({0}, '{1}', {2}, {3}, '{4}')"
          insert_query = insert_query.format(dataframe["RANK"][i],dataframe["USERNAME"][i],dataframe["ACCURACY"][i],dataframe["ENTRIES"][i],dataframe["LAST"][i])
          print(insert_query)
          curs = conn.cursor()
          curs.execute(insert_query)
          # # curs.commit()
          load_dataset()
          curs.close()
          conn.close()
          print("YOUR SUBMISSION SAVED SUCCESSFULLY >>>>>>>>>>>>")


def update_record(dataframe):
  print("IN update_record >>>>>>>>")
  for i in range(0, len(dataframe)):
      if dataframe["USERNAME"][i] == FLAGS.username:
          # print(submissions_df.iloc[i])
          update_query = "UPDATE "+ (dataSet['schema'] + '.' if (len(dataSet['schema'].strip()) != 0) else '') + dataSet['table']
          update_query = update_query +" SET RANK = {0}, ENTRIES = {1}, ACCURACY = {2}, LAST = '{3}' WHERE USERNAME = '{4}'"
          update_query = update_query.format(dataframe["RANK"][i], dataframe["ENTRIES"][i], dataframe["ACCURACY"][i], dataframe["LAST"][i], dataframe["USERNAME"][i])
          print(update_query)
          curs = conn.cursor()
          curs.execute(update_query)
          # # curs.commit()
          load_dataset()
          curs.close()
          conn.close()
          print("YOUR SUBMISSION UPDATED SUCCESSFULLY >>>>>>>>>>>>")

def check_accuracy():
    user_df = pd.read_csv(DATA_DIR+'/'+FLAGS.username+'_test_results.csv', header=0, delimiter=",")
    main_df = pd.read_csv(DATA_DIR+'/'+FLAGS.answers_file_path, header=0, delimiter=",")
    matches_found = 0
    miss_matches = 0
    for i in range(0, len(main_df)):
        if main_df.iloc[i, -1] == user_df.iloc[i,-1]:
            matches_found = matches_found + 1
        else:
            miss_matches = miss_matches + 1

    accuracy = matches_found/len(main_df) * 100
    print("matches_found: >> ", matches_found)
    print("miss_matched: >> ", miss_matches)
    print("accuracy: >>> ", accuracy)
    global submissions_df
    user_record_found = False
    columns_seq = ["RANK", "USERNAME", "ACCURACY", "ENTRIES", "LAST"]
    if submissions_df.empty:
        print("submissions dataset does not exists ... ")
        details = {"RANK": 1, "USERNAME": FLAGS.username, "ACCURACY": accuracy, "ENTRIES": 1, "LAST": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        submissions_df = pd.DataFrame(details, index=[0])
        submissions_df = details_df.round(2)
    else:
        print("submissions dataset exists ... ")
        for i in range(0, len(submissions_df)):
            if submissions_df["USERNAME"][i] == FLAGS.username:
                user_record_found = True
                submissions_df["ACCURACY"][i] = accuracy
                submissions_df["ENTRIES"][i] = submissions_df["ENTRIES"][i] + 1
                submissions_df["LAST"][i] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if user_record_found == False:
            print("User record not found ... ")
            details = {"RANK": 0, "USERNAME": FLAGS.username, "ACCURACY": accuracy, "ENTRIES": 1, "LAST": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            submissions_df = submissions_df.append(details, ignore_index=True)

        submissions_df = submissions_df.round(2)
        submissions_df['RANK'] = submissions_df['ACCURACY'].rank(ascending=False)
        submissions_df = submissions_df.sort_values(by='RANK', ascending=True)
        submissions_df["RANK"] = submissions_df["RANK"].astype(int)
    if user_record_found == False:
       save_record(submissions_df)
    else:
       update_record(submissions_df)
    # submissions_df.to_csv(submissions_file, index=False, columns=columns_seq)

def main():
    if FLAGS.username is None:
        FLAGS.username = getpass.getuser()

    init_steps()
    check_accuracy()
    # submissions_df = pd.read_csv(submissions_file, delimiter=",")
    print("\n\n<<<<<<<<<< RESULTS >>>>>>>>>>>>>")
    print(submissions_df.head())
    # for i in range(0, len(submissions_df)):
      # print("Rank: ", submissions_df[i]["rank"], "Username: ", submissions_df[i]["username"], ", Score: ", submissions_df[i]["accuracy"], ", Entries: ", submissions_df[i]["entries"], ", Last: ", submissions_df[i]["last"])

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--answers_file_path', default='Titanictrain -- Blinddataset -- WithScoring.csv', type=str, help='Answers File Path')
  parser.add_argument('--username', type=str, help='Username')

  FLAGS, unparsed = parser.parse_known_args()
  main()
