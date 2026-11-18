import random
from locust import HttpUser, task, between

random.seed(42)

with open("sentences.txt", encoding="utf-8") as f:
    SENTENCES = [line.strip() for line in f if line.strip()]


class SentimentUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict(self):
        text = random.choice(SENTENCES)
        with self.client.post(
            "/predict",
            json={"text": text},
            timeout=30,
            catch_response=True,
            name="/predict",
        ) as r:
            if r.status_code != 200:
                r.failure(f"HTTP {r.status_code}")
