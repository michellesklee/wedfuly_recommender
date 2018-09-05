from flask import Flask, render_template, request
import json
import requests
import socket
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__, static_url_path='/static')

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/recommendations', methods=['GET', 'POST'])
def recommendations():
    return render_template('recommender.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
