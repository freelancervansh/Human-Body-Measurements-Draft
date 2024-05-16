import mysql.connector


def add_to_rds():
  mydb = mysql.connector.connect(
    host="mysqltest.csprg633a9rz.us-east-1.rds.amazonaws.com",
    user="root",
    password="",
    database="shopnow_uat"
  )

  mycursor = mydb.cursor()

  sql = "INSERT INTO contact_us (name, email) VALUES (%s, %s)"
  val = ("John", "test@john.com")
  mycursor.execute(sql, val)

  mydb.commit()

  print(mycursor.rowcount, "record inserted.")


def add_to_s3():
  BUCKET_NAME = ''
  HOST_URL = ""
  s3 = boto3.resource('s3', 
                      aws_access_key_id="xxxxxxx",
                      aws_secret_access_key="xxxxxxx"
                  )
  bucket = s3.Bucket(BUCKET_NAME)