from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from chatbot import get_answer

app = FastAPI(title="College Helpdesk Chatbot")

templates = Jinja2Templates(directory="templates")


class Question(BaseModel):
    question: str


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.post("/chat")
def chat(data: Question):
    answer = get_answer(data.question)
    return {"answer": answer}