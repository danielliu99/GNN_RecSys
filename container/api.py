from fastapi import FastAPI, Request
import torch 


# Instantiating FastAPI
api = FastAPI()

# Defining a test root path and message
@api.get('/')
def root():
  return {'message': 'Hello friends!'}

# Load models or embeddings
user_embedding = torch.load("../weights/user_embedding.pt")
item_embedding = torch.load("../weights/item_embedding.pt") 



# Defining the prediction endpoint without data validation
@api.post('/predict')
async def predict(request: Request):

    # Getting the JSON from the body of the request
    input_data = await request.json()

    # Converting JSON to Pandas DataFrame
    # input_df = pd.DataFrame([input_data])
    query_user_id = input_data['user_id']
    query_user_embedding = user_embedding[query_user_id]
    relevance_score = torch.matmul(query_user_embedding, torch.transpose(item_embedding, 0, 1))
    topk_relevance_indices = torch.topk(relevance_score, 10).indices
    pred = topk_relevance_indices.cpu().numpy().tolist()

    # Getting the prediction from the Logistic Regression model
    # pred = lr_model.predict(input_df)[0]

    return pred 
