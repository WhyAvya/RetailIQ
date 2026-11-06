from fastapi import FastAPI

app = FastAPI(title="RetailIQ API")

@app.get("/")
def home():
    return {"message": "RetailIQ API is running successfully!"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}