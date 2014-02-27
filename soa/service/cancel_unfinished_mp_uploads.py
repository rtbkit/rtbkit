"""
Cancels multipart upload that are more that have been 
initiated more than 24h ago.
"""
import argparse
from boto.s3.connection import S3Connection
from datetime import datetime, timedelta

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--id", help="S3 key id", required=True)
parser.add_argument("-k", "--key", help="S3 key", required=True)
args = parser.parse_args()

conn = S3Connection(args.id, args.key)
now = datetime.now()
for bucket in conn.get_all_buckets():
    for mp in bucket.list_multipart_uploads():
        initiated = datetime.strptime(mp.initiated[0:-5], "%Y-%m-%dT%H:%M:%S")
        expires = initiated + timedelta(days=1)
        if expires < now:
            print "Canceling upload for {0}/{1}. Expires: {2}" \
                .format(bucket.name, mp.key_name, expires)
            try:
                mp.cancel_upload()
            except Exception as e:
                print "-" + str(e)
        else:
            print "Keeping {0}/{1}. Expires: {2}" \
                .format(bucket.name, mp.key_name, expires)

