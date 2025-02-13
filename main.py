from fastapi import FastAPI, Request
import actions, awimlib, metadata_tools

app = FastAPI()

@app.post("/whatsinphoto/")
async def whatsinphoto(request: Request):
    awimTag = await request.json()
    
    return {'message': 'Hello World'}