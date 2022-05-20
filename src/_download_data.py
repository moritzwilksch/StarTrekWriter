import io

import boto3
from rich.console import Console

c = Console()

s3 = boto3.resource("s3")
bucket = s3.Bucket("deep-text-generation")

bucket.download_file("StarTrek/all_scripts_raw.json", "data/all_scripts_raw.json")
c.print("[bold green][INFO][/] Downloaded all_scripts_raw.json from S3.")
