# -------------------------
# Submit to server
# -------------------------
import torch
import requests
# model = torch.jit.load("model_v2.pt", map_location="cuda:0")
# model.eval()
# model = model.to(dtype=torch.float32)
# torch.jit.save(model, "model_v2_fp32.pt")

def submit_model(token: str, model_path: str, server_url="http://hadi.cs.virginia.edu:9000"):
    with open(model_path, "rb") as f:
        files = {"file": f}
        data = {"token": token}
        response = requests.post(f"{server_url}/submit", data=data, files=files)
        resp_json = response.json()
        if "message" in resp_json:
            print(f"✅ {resp_json['message']}")
        else:
            print(f"❌ Submission failed: {resp_json.get('error')}")

my_token="79df3806ff358c68486be939f8bd2196"
submit_model(my_token, "model_v2_fp32.pt")