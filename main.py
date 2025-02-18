import uvicorn
from fastapi import FastAPI, Request
import json
import actions, awimlib, metadata_tools

app = FastAPI()

@app.post('/testpost')
async def test(request: Request):
    try:
        whatsthis = request.body()
        print(whatsthis)
        data_received = json.loads(await request.body())
        print(data_received)
        awimTag = await request.json()
        print(awimTag)
    except:
        print('some error on the post request attempt')


@app.get("/testget")
async def test(request: Request):
    try:
        data_received_dict = request.query_params._dict
        print(data_received_dict['foo'])
        print(data_received_dict['foo2'])
        data_received = request.query_params.get('foo')
        print(data_received)
        data_received = request.query_params.get('foo2')
        print(data_received)
        # this works with http://localhost:8080/testget?foo=bar&foo2=bar2
    except:
        print('some error on the get request')
    
    return data_received_dict

if __name__ == '__main__':
    uvicorn.run(app, port=8080)