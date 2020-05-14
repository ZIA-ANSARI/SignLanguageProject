import pandas as pd
from flask import Flask, jsonify, request
import pickle
import numpy as np

# load model
model = pickle.load(open('/home/machine/finalized_model_xgboost_a_c_h_y.sav','rb'))

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
	data = request.get_json(force=True)
	print(data)

	# convert data into dataframe
	data.update((x, [y]) for x, y in data.items())
	data_df = pd.DataFrame.from_dict(data)
	mul = np.array([1.30442204e+00, 1.57782492e+00, 8.68026471e+01, 1.45386853e+00,
 1.37637777e+00, 1.53220014e-02, 1.27661361e+00, 1.27090696e+00,
 8.47906730e-03, 1.25045517e+00, 1.24732721e+00, 6.38467678e-03,
 1.26509795e+00, 1.11273370e+00, 5.16129032e-03, 1.53347393e+00,
 1.29205492e+00, 5.90405904e-03, 1.42389006e+00,1.15681900e+00,
 4.25305688e-03, 1.36571435e+00, 1.13823540e+00, 3.82226469e-03,
 1.26956745e+00, 1.10919929e+00, 3.63388599e-03, 1.49756311e+00,
 1.37481134e+00, 6.16332820e-03,1.40907927e+00, 1.19492225e+00,
 4.39923013e-03, 1.35020663e+00, 1.23810000e+00, 4.32315590e-03,
 1.21406970e+00, 1.25812522e+00, 4.20831142e-03, 1.47559308e+00,
 1.40786955e+00, 6.37704265e-03, 1.47524646e+00, 1.26727250e+00,
 4.79616307e-03, 1.52609104e+00, 1.29996525e+00, 5.10204082e-03,
 1.52396530e+00, 1.29676188e+00, 5.24934383e-03, 1.45633118e+00,
 1.42966173e+00, 6.46464646e-03,1.43261856e+00,1.34402914e+00,
 5.18638574e-03, 1.32768136e+00, 1.37856719e+00, 5.22875817e-03,
 1.22937186e+00, 1.34884949e+00, 5.11508951e-03])
	
	data_df.multiply(mul, axis=1)





	# predictions
	result = model.predict(data_df)

	# send back to browser
	output = {'results': int(result[0])}

	# return data
	return str(result[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
