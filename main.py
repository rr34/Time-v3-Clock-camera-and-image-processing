import uvicorn
from fastapi import FastAPI, Request
import json
import actions, awimlib, metadata_tools

app = FastAPI()

@app.post("/test/")
async def test(request: Request):
    try:
        data_received = json.loads(await request.body())
        print(data_received)
        awimTag = await request.json()
        print(awimTag)
    except:
        print('some error')
    
    return {'message': 'Hello World'}

if __name__ == '__main__':
    uvicorn.run(app, port=8080)