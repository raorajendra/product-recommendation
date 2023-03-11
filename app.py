from flask import Flask,jsonify,request
import requests
import json
from flask_cors import CORS 
from model import get_top_5_recommondation

app=Flask(__name__)
CORS(app)


import warnings
warnings.filterwarnings("ignore")
@app.route('/hello')

def hello_world():
    return 'hello world'
    
    
@app.route('/getProductReccomendation',methods = ['GET'])    
def get_top_n_recommendation():
    args = request.args
    user=args.get("user", default="", type=str)
    
    if not user:
        error = {'error': 'User ID is required.'}
        return jsonify(error), 400
    
    #recommendation =get_product_recommendation(user)
    recommendation = get_top_5_recommondation(user)
    print(recommendation)    
    dict_recommand={'recommand': recommendation['product'].tolist()}
    print('output as dictionary::',dict_recommand)
    return json.dumps(dict_recommand)
    
    
def get_product_recommendation(user):
    # Your logic here to generate the recommendation for the given user and n
    
    # For this example, we'll just return a hardcoded recommendation
    recommendation = ["user1", "user2", "user3"]
    
    # Return the recommendation as a dictionary
    response = {"recommand": recommendation}
    return response
    
    
    


if __name__=='__main__':
   app.run(host='127.0.0.1', port=5000, debug=False)

