import subprocess
import boto3
import botocore
import os
import sys
import json
import time
import io
import glob
import shutil

s3 = boto3.resource('s3')
BUCKET_NAME = 'oirml'
outputdir = '/root/oir/output'

cluster_name = 'default'
ecs_client = boto3.client('ecs')
ec2_client = boto3.client('ec2')

def uploadResult(localfile, s3key):
  data = open('%s/%s' % (outputdir, localfile), 'rb')
  s3.Bucket(BUCKET_NAME).put_object(Key=s3key, Body=data)

def process(sourceKey):
  if not os.path.exists(outputdir):
    os.makedirs(outputdir)

  ## get key of the source file at s3
  sourceFileName = os.path.basename(sourceKey)
  basename, ext = os.path.splitext(sourceFileName)
  bucketDirname = os.path.dirname(sourceKey)

  ## download input image from s3
  inputFile = '%s/%s' % (outputdir, sourceFileName) 
  s3.Bucket(BUCKET_NAME).download_file(sourceKey, inputFile)

  if 'tif' in ext:
    subprocess.call(['convert', inputFile, '%s/input.png' % outputdir])
    files = glob.glob('%s/input*.png' % outputdir)
    if len(files) > 1:
      inputFile = files[0]

  ## inference
  output = subprocess.check_output(['/root/torch/install/bin/th', '/root/oir/predict.lua', '--imageFile', inputFile])
  lines = output.splitlines()
  status = lines[len(lines)-1]
  
  if 'true' in status:
    ## upload results to s3
    uploadResult('quantified.png', '%s/%s_quantified.done.png' % (bucketDirname, basename))
    uploadResult('retina.png', '%s/%s_retina.done.png' % (bucketDirname, basename))

def delete_ecs_instance():
  instanceId = subprocess.check_output(['curl', 'http://169.254.169.254/latest/meta-data/instance-id'])
  ec2_termination_resp = ec2_client.terminate_instances(
    DryRun=False,
    InstanceIds=[
      instanceId,
    ]
  ) 

sqs = boto3.client('sqs')
QUEUE_URL = "https://us-west-2.queue.amazonaws.com/686306465972/oir2process.fifo"
MAX_RETRIES = 5
retries = 0

try:
  while True:
    response = sqs.receive_message(
      QueueUrl=QUEUE_URL, 
      VisibilityTimeout=123,
      MessageAttributeNames=['All'])
    if 'Messages' in response:
      retries = 0
      messages = response['Messages']
      for message in messages:
        body = json.loads(message['Body'])
        key = body['Records'][0]['s3']['object']['key']
        print(key)
        haserr = False
        try:
          process(key)
        except botocore.exceptions.ClientError as e1:
          errmsg = {'error': 'S3 I/O failed', 'errordetail': e1.response['Error']['Code']}
          haserr = True
        except Exception as e:
          errmsg = {'error': 'unexpected error', 'errordetail': str(e)}
          haserr = True
        finally:
          if haserr:
            with open('oir/output/ratio.json', 'w') as f:
              json.dump(errmsg, f)
          sourceFileName = os.path.basename(key)
          basename = os.path.splitext(sourceFileName)[0]
          bucketDirname = os.path.dirname(key)
          uploadResult('ratio.json', '%s/%s_ratio.done.json' % (bucketDirname, basename))
          
          ## clean up: delete all the input and outputs
          shutil.rmtree(outputdir)
        
          # Delete received message from queue
          receipt_handle = message['ReceiptHandle']
          sqs.delete_message(
            QueueUrl=QUEUE_URL,
            ReceiptHandle=receipt_handle  
          )
    else:
      if retries == MAX_RETRIES:
        break;
      time.sleep(5)
      retries = retries + 1
finally:
  print("finally")
  delete_ecs_instance()
