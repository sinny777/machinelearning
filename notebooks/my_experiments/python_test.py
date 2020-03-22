import pandas as pd
import re

def extract_username():
    regexStr = r'^([^@]+)@[^@]+$'
    username = 'arjun.katyal@cognizant.com'
    matchobj = re.search(regexStr, username)
    if not matchobj is None:
        print(matchobj.group(1))
    else:
        print(username)

def test_run():
    try:
        df = pd.read_csv('data/ICP4DUSERS_2.csv', header=0, delimiter=",")
        # user_data = dataframe.loc[dataframe["USERNAME"] == 'gurvsin3']
        df.loc[df['USERNAME']== 'gurvsin3', ['NAME']] = 'Gurvinder Singh'
        user_data = df.loc[df["USERNAME"] == 'gurvsin3']
        print(user_data)
    except FileNotFoundError:
        print("YOUR RESULTS FILE NOT FOUND")

test_run()
