from fastapi import FastAPI
from pydantic import BaseModel
import json

app = FastAPI()

# Load catalog
with open("catalog.json", "r") as f:
    catalog = json.load(f)

# Request body
class ChatRequest(BaseModel):
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):

    user_message = req.message.lower()

    recommendations = []

    # Simple retrieval
    for item in catalog:

        text = (
            item["name"] + " " +
            item["description"]
        ).lower()

        if any(word in text for word in user_message.split()):
            recommendations.append(item)

    # Clarifying question
    if len(recommendations) == 0:
        return {
            "response": (
                "Could you specify the skill area? "
                "Example: Python, Java, SQL, Cognitive."
            ),
            "recommendations": []
        }

    return {
        "response": "Here are recommended SHL assessments.",
        "recommendations": recommendations[:3]
    }