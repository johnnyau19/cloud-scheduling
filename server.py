from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np


app = FastAPI()


class Observation(BaseModel):
    server0_est_workload: float
    server1_est_workload: float
    server2_est_workload: float
    new_job_est_workload: float


ort_session = ort.InferenceSession("model/best_model.onnx")


@app.get("/", response_class=HTMLResponse)
def home():
    return "<h1>Cloud Scheduler</h1>"


@app.post("/predict")
def predict(observation: Observation):
    obs = np.array(list(observation.model_dump().values()), dtype=np.float32)
    obs = obs.reshape(1, -1)
    action = ort_session.run(None, {"input": obs})[0]
    return {"action": int(action)}
