
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
    set_username()

def extract_username(username):
    regexStr = r'^([^@]+)@[^@]+$'
    matchobj = re.search(regexStr, username)
    if not matchobj is None:
        return matchobj.group(1)
    else:
        return username

def set_username():
    FLAGS.username = getpass.getuser()
    users_dataset = dsx_core_utils.get_remote_data_set_info(FLAGS.users_dataset)
    users_df = load_dataset(users_dataset)
    for i in range(0, len(users_df)):
        if str(users_df["USER_ID"][i]) == FLAGS.username:
           # print(users_df.iloc[i])
           FLAGS.username = extract_username(users_df["USERNAME"][i])
           FLAGS.name = extract_username(users_df["NAME"][i])
           # print("Username Type: ", type(FLAGS.username), ", Username: ", FLAGS.username)

def get_submissions_dataframe():
    submissions_dataset = dsx_core_utils.get_remote_data_set_info(FLAGS.submissions_dataset)
    submissions_df = load_dataset(submissions_dataset)
    return submissions_df

def load_dataset(dataSet):
  global conn
  try:
      conn
  except NameError:
      dataSource = dsx_core_utils.get_data_source_info(dataSet['datasource'])
      if (sys.version_info >= (3, 0)):
      	conn = jaydebeapi.connect(dataSource['driver_class'], dataSource['URL'], [dataSource['user'], dataSource['password']])
      else:
      	conn = jaydebeapi.connect(dataSource['driver_class'], [dataSource['URL'], dataSource['user'], dataSource['password']])

  query = 'SELECT * FROM ' + (dataSet['schema'] + '.' if (len(dataSet['schema'].strip()) != 0) else '') + dataSet['table']
  dataframe = pd.read_sql(query, con=conn)
  return dataframe

def save_record(dataframe):
  print("IN save_record >>>>>>>>")
  for i in range(0, len(dataframe)):
      if dataframe["USERNAME"][i] == FLAGS.username:
          dataSet = dsx_core_utils.get_remote_data_set_info(FLAGS.submissions_dataset)
          insert_query = "INSERT INTO "+ (dataSet['schema'] + '.' if (len(dataSet['schema'].strip()) != 0) else '') + dataSet['table']
          insert_query = insert_query + "(USERNAME, NAME, ACCURACY, ENTRIES, LAST) VALUES ('{0}', '{1}', {2}, {3}, '{4}')"
          insert_query = insert_query.format(dataframe["USERNAME"][i], dataframe["NAME"][i],dataframe["ACCURACY"][i],dataframe["ENTRIES"][i],dataframe["LAST"][i])
          curs = conn.cursor()
          curs.execute(insert_query)
          # load_dataset(dataSet)
          curs.close()
          print("YOUR SUBMISSION SAVED SUCCESSFULLY >>>>>>>>>>>>")


def update_record(dataframe):
  print("IN update_record >>>>>>>>")
  for i in range(0, len(dataframe)):
      if dataframe["USERNAME"][i] == FLAGS.username:
          # print(submissions_df.iloc[i])
          dataSet = dsx_core_utils.get_remote_data_set_info(FLAGS.submissions_dataset)
          update_query = "UPDATE "+ (dataSet['schema'] + '.' if (len(dataSet['schema'].strip()) != 0) else '') + dataSet['table']
          update_query = update_query +" SET ENTRIES = {0}, ACCURACY = {1}, LAST = '{2}' WHERE USERNAME = '{3}'"
          update_query = update_query.format(dataframe["ENTRIES"][i], dataframe["ACCURACY"][i], dataframe["LAST"][i], dataframe["USERNAME"][i])
          curs = conn.cursor()
          curs.execute(update_query)
          # load_dataset(dataSet)
          curs.close()
          print("YOUR SUBMISSION UPDATED SUCCESSFULLY >>>>>>>>>>>>")

def check_accuracy():
    try:
        user_df = pd.read_csv(DATA_DIR+'/'+FLAGS.username+FLAGS.user_file_suffix, header=0, delimiter=",")
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
        # global submissions_df
        submissions_df = get_submissions_dataframe()
        user_record_found = False
        columns_seq = ["USERNAME", "NAME", "ACCURACY", "ENTRIES", "LAST"]
        if submissions_df.empty:
            print("New Submission Entry for User: ", FLAGS.username)
            details = {"USERNAME": FLAGS.username, "NAME": FLAGS.name, "ACCURACY": accuracy, "ENTRIES": 1, "LAST": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            submissions_df = pd.DataFrame(details, index=[0])
            submissions_df = submissions_df.round(2)
        else:
            # print("submissions dataset exists ... ")
            for i in range(0, len(submissions_df)):
                if submissions_df["USERNAME"][i] == FLAGS.username:
                    print("New Submission Entry for User: ", FLAGS.username)
                    user_record_found = True
                    submissions_df.loc[submissions_df['USERNAME'] == FLAGS.username, ['ACCURACY']] = accuracy
                    submissions_df.loc[submissions_df['USERNAME'] == FLAGS.username, ['ENTRIES']] = submissions_df["ENTRIES"][i] + 1
                    submissions_df.loc[submissions_df['USERNAME'] == FLAGS.username, ['LAST']] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if user_record_found == False:
                print("New User's Submission: ", FLAGS.username)
                details = {"USERNAME": FLAGS.username, "NAME": FLAGS.name, "ACCURACY": accuracy, "ENTRIES": 1, "LAST": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                submissions_df = submissions_df.append(details, ignore_index=True)

            submissions_df = submissions_df.round(2)

        if user_record_found == False:
           save_record(submissions_df)
        else:
           update_record(submissions_df)
        print("\n\n<<<<<<<<<< YOU CAN CHECK YOUR RANK IN THE ANALYTICS LEADERBOARD >>>>>>>>>>>>>")
    except FileNotFoundError:
        print("YOUR RESULTS FILE NOT FOUND")
        print("Please make sure that you save your results csv file in the required format.")

def main():
    try:
        init_steps()
        check_accuracy()
        # for i in range(0, len(submissions_df)):
          # print("Rank: ", submissions_df[i]["rank"], "Username: ", submissions_df[i]["username"], ", Score: ", submissions_df[i]["accuracy"], ", Entries: ", submissions_df[i]["entries"], ", Last: ", submissions_df[i]["last"])
    finally:
        try:
            conn
            if conn:
              conn.close()
              # print("CONECTION Closed Finally ...")
        except NameError:
            print("......")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--answers_file_path', default='Titanictrain -- Blinddataset -- WithScoring.csv', type=str, help='Answers File Path')
  parser.add_argument('--username', type=str, help='Username')
  parser.add_argument('--user_file_suffix', type=str, default='_test_results.csv', help='Username')
  parser.add_argument('--users_dataset', type=str, default='ICP4DUSERS', help='ICP4DUSERS Dataset')
  parser.add_argument('--submissions_dataset', type=str, default='SUBMISSIONS', help='SUBMISSIONS Dataset')


  FLAGS, unparsed = parser.parse_known_args()
  main()
