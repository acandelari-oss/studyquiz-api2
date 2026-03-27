from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this as necessary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Your other FastAPI routes here
