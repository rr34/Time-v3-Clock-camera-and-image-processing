import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import json
import actions, awimlib, metadata_tools

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


@app.post('/testpost')
async def testpost(request: Request):
    print('get here?')
    try:
        whatsthis = await request.body()
        print(whatsthis)
        data_received = json.loads(request.body())
        print(data_received)
        awimTag = await request.json()
        print(awimTag)
    except:
        print('some error on the post request attempt')


# @app.get("/testget/")
# async def testget(request: Request):
#     try:
#         data_received_dict = request.query_params._dict
#         print(data_received_dict['foo'])
#         print(data_received_dict['foo2'])
#         data_received = request.query_params.get('foo')
#         print(data_received)
#         data_received = request.query_params.get('foo2')
#         print(data_received)
#         # this works with http://localhost:8000/testget?foo=bar&foo2=bar2
#     except:
#         print('some error on the get request')
    
#     return data_received_dict

if __name__ == '__main__':
    uvicorn.run(app)