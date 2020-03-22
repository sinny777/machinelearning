
# python publish.py --model_path "gurvsin3_model.pkl" --framework_name "scikit-learn"

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

RESULT = {}

FLAGS = None

DATA_DIR = os.environ['DSX_PROJECT_DIR']+'/datasets'
# DATA_DIR = 'data'

submissions_file = DATA_DIR+'/submissions.csv'

def init_steps():
    global submissions_df
    if(os.path.exists(submissions_file)):
        submissions_df = pd.read_csv(submissions_file, delimiter=",")
        submissions_df = submissions_df.sort_values(by='accuracy', ascending=False)
    else:
        submissions_df = pd.DataFrame()

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
    columns_seq = ["rank", "username", "accuracy", "entries", "last"]
    if submissions_df.empty:
        print("submission.csv file does not exists ... ")
        details = {"rank": 1, "username": FLAGS.username, "accuracy": accuracy, "entries": 1, "last": datetime.now()}
        details_df = pd.DataFrame(details, index=[0])
        # print(details_df.head())
        details_df = details_df.round(2)
        details_df.to_csv(submissions_file, index=False, columns=columns_seq)
    else:
        print("submission.csv file exists ... ")
        user_record_found = False
        for i in range(0, len(submissions_df)):
            if submissions_df["username"][i] == FLAGS.username:
                user_record_found = True
                submissions_df["accuracy"][i] = accuracy
                submissions_df["entries"][i] = submissions_df["entries"][i] + 1
                submissions_df["last"][i] = datetime.now()

        if user_record_found == False:
            print("User record not found ... ")
            details = {"rank": 0, "username": FLAGS.username, "accuracy": accuracy, "entries": 1, "last": datetime.now()}
            submissions_df = submissions_df.append(details, ignore_index=True)

        submissions_df = submissions_df.round(2)
        submissions_df['rank'] = submissions_df['accuracy'].rank(ascending=False)
        submissions_df = submissions_df.sort_values(by='rank', ascending=True)
        submissions_df["rank"] = submissions_df["rank"].astype(int)
        submissions_df.to_csv(submissions_file, index=False, columns=columns_seq)

def main():
    # FLAGS.username = getpass.getuser()
    init_steps()
    check_accuracy()
    submissions_df = pd.read_csv(submissions_file, delimiter=",")
    print("\n\n<<<<<<<<<< RESULTS >>>>>>>>>>>>>")
    print(submissions_df.head())
    # for i in range(0, len(submissions_df)):
      # print("Rank: ", submissions_df[i]["rank"], "Username: ", submissions_df[i]["username"], ", Score: ", submissions_df[i]["accuracy"], ", Entries: ", submissions_df[i]["entries"], ", Last: ", submissions_df[i]["last"])

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--answers_file_path', default='Titanictrain -- Blinddataset -- WithScoring.csv', type=str, help='Answers File Path')
  parser.add_argument('--username', type=str, help='Username')
  parser.add_argument('--model_uid', type=str, help='Model UID')

  FLAGS, unparsed = parser.parse_known_args()
  main()
