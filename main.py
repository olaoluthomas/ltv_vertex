# -*- coding: utf-8 -*-

import os
from flask import Flask
from src.trainer import run_training
from src.scorer import run_model

t = 12
d = 0.00764

app = Flask(__name__)


@app.route("/")
def train():
    """
    Train model to generate shape and scale parameters for MBG & GGF.
    """
    if os.path.exists("./data/train_data.csv") == True:
        run_training(train_data="./data/train_data.csv")


# def score():
#     """
#     Run saved parameters on population file and output to csv.
#     """
#     # how would the output be dumped to CS/BQ?
#     if os.path.exists("mbg.pkl") == True & os.path.exists(
#             "ggf.pkl") == True & os.path.exists(
#                 "./data/score_data.csv") == True:
#         run_model(input_file="./data/score_data.csv", t=t, r=d)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))