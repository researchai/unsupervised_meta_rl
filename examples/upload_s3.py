# THIS DATA IS *SECRET*
# **DO NOT SHARE**
# TODO Move data to public S3
import boto3
import os
import subprocess
import time

files = os.listdir()
wait_time = 1200.0
start_ind = int(input("start index: "))
num_files = int(input("num_files: "))
s3_name = input("s3_name: ")
print(files)
while 1:
    for f, i in zip(files, range(start_ind, start_ind+num_files)):
        subprocess.run("aws s3 sync {file}/ {s3_name}_{i}".format(file=f, s3_name=s3_name, i=i), shell=True)
    time.sleep(wait_time)
