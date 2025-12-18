from openai import OpenAI
client = OpenAI()

# List batches and print only their IDs (one per line)
resp = client.batches.list(limit=10)

for batch in getattr(resp, "data", resp):
    print(batch)
    # print out id, "request_counts.completed" and "request_counts.total"
    # print(getattr(batch, "id", None), " ", getattr(batch, "request_counts", None))