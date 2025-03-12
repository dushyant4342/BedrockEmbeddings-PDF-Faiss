import pymysql
import boto3
import json

def get_secret():
    secret_name = "rds!db-04ec150d-7199-42ce-a8e4-8395c7a5f259"
    region_name = "ap-south-1"
    
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    response = client.get_secret_value(SecretId=secret_name)
    return json.loads(response["SecretString"])  # Convert JSON string to dictionary

# Get credentials
creds = get_secret()
host = "mysql-rds-webapp.clo4m2e8y7j9.ap-south-1.rds.amazonaws.com"  # Make sure 'host' is stored in Secrets Manager
user = creds["username"]

password = creds["password"]
print("password:",password)

db = "mysql-rds-webapp"  # Replace with your DB name


# Connect to MySQL
conn = pymysql.connect(host=host, user=user, password=password, database=db, port=3306)

# Query Example
cursor = conn.cursor()
cursor.execute("SHOW TABLES;")
tables = cursor.fetchall()
print("Tables:", tables)

conn.close()
