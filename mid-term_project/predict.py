import pickle

from flask import Flask
from flask import request
from flask import jsonify


def load(filename: str):
    with open(filename, 'rb') as f_in:
        return pickle.load(f_in)


dv = load('/home/administrator/mlzoomcamp/project/dv.bin')
model = load('/home/administrator/mlzoomcamp/project/model.bin')

app = Flask('wine_q')


@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict(X,dv,model)
    #get_card = y_pred >= 0.5

    result = {
        'get_probability': float(y_pred),
       # 'get_card': bool(get_card)
    }

    return jsonify(result)


#@app.route('/test',methods = ['GET'])

#def test():
    #url = "http://localhost:9697/predict"
    #client = {"	fixed_acidity": 7.4, "residual_sugar": 1.9, "density": 0.9978}
    #response = requests.post(url, json=client).json()
    #print(response)
    #return 'test'




if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9697)