# load testing

from locust import HttpUser, task, between
import json
import numpy as np

class HARModelUser(HttpUser):
    wait_time = between(1, 3)

    @task(1)
    def predict_activity(self):
        """Test HAR model prediction endpoint"""
        headers = {'Content-Type': 'application/json'}
        
        # Sample data for HAR dataset 
        payload = {
            "data": [[
                # Adjust these features according to your HAR dataset columns
                np.random.uniform(-1, 1),  # Feature 1
                np.random.uniform(-1, 1),  # Feature 2
                np.random.uniform(-1, 1),  # Feature 3
                np.random.uniform(-1, 1),  # Feature 4
                np.random.uniform(-1, 1),  # Feature 5
                np.random.uniform(-1, 1)   # Feature 6
            ]]
        }

        with self.client.post(
            "/predict",
            json=payload,
            headers=headers,
            catch_response=True
        ) as response:
            try:
                if response.status_code == 200:
                    result = response.json()
                    if "prediction" in result:
                        response.success()
                    else:
                        response.failure("Missing prediction in response")
                else:
                    response.failure(f"HTTP {response.status_code}")
            except Exception as e:
                response.failure(str(e))

    @task(2)
    def health_check(self):
        """Test health check endpoint"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: HTTP {response.status_code}")