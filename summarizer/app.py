from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from extractive_summary import extractive_summarization

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Abstractive summarization model
model = AutoModelForSeq2SeqLM.from_pretrained('saved_model', local_files_only=True, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('saved_model', local_files_only=True)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "summary": "", "text": "", "error_message": None})

@app.post("/summarize", response_class=HTMLResponse)
async def summarize(request: Request, text: str = Form(None), summary_type: str = Form(...)):
    if not text:
        return templates.TemplateResponse("index.html", {"request": request, "summary": "", "text": "", "error_message": "No text input provided."})
    summary = ""
    if summary_type == "abstractive":
        summary = abstractive_summarization(text)
    else:
        summary = extractive_summarization(text)
    return templates.TemplateResponse("index.html", {"request": request, "summary": summary, "text": text, "error_message": None})

def abstractive_summarization(text):
    tokens = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(tokens, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
