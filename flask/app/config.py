from os import environ

from dotenv import load_dotenv
load_dotenv()


class Config:
    ALLOWED_IP = environ.get('ALLOWED_IP')
    S3_ACCESS_KEY = environ['S3_ACCESS_KEY']
    S3_SECRET_ACCESS_KEY = environ['S3_SECRET_ACCESS_KEY']
    S3_BUCKET_NAME = environ["S3_BUCKET_NAME"]