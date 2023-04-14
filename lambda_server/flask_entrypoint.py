from flask import Flask
from flask import request
from flask import json

app = Flask(__name__)

#global: to be initialized in initial_setup - a common storage variable for everything, including initiated models and current states of each of the users.
STORAGE = {}


def initial_setup():
  """
  Initial setup for the server
    - Load all the models.
    - Load context pairs and be ready for answering the requests.
  """
  pass


@app.route('/setup_new_user/', methods=['POST'])
def setup_new_user():
  """
  Setup for a new user - build the connection.
   - return a new (model, context, random id)
   - save the model, context and random id in the storage.
   - mark this triplet as used.
  """

  payload = json.loads(request.get_data().decode('utf-8'))
  
  data = {}
  data["relayed"] = payload # just for debugging

  return json.dumps(data)


@app.route('/model_resp/', methods=['POST'])
def model_resp():
  """
  Get model response when the user has either just started or has already been talking to the model.
  """

  payload = json.loads(request.get_data().decode('utf-8'))
  
  data = {}
  data["relayed"] = payload # just for debugging

  return json.dumps(data)


@app.route('/model_resp_post_sd/', methods=['POST'])
def model_resp_post_sd():
  """
  Get model response when the user has submitted a deal.
  """

  payload = json.loads(request.get_data().decode('utf-8'))
  
  data = {}
  data["relayed"] = payload # just for debugging

  return json.dumps(data)


@app.route('/model_resp_post_wa/', methods=['POST'])
def model_resp_post_wa():
  """
  Get model response when the user has walked away.
  """

  payload = json.loads(request.get_data().decode('utf-8'))
  
  data = {}
  data["relayed"] = payload # just for debugging

  return json.dumps(data)


@app.route('/reset_internally/', methods=['POST'])
def reset_internally():
  """
  Reset the internal states, as if no user has been connected to the server yet after the server has been started. You can still keep the models loaded.
  """

  payload = json.loads(request.get_data().decode('utf-8'))
  
  data = {}
  data["relayed"] = payload # just for debugging

  return json.dumps(data)


@app.route('/report_stats/', methods=['POST'])
def report_stats():
  """
  Report basic stats about the current state of the server.
    - # of models, contexts, etc.
    - how many users have connected so far.
    - how many users are remaining, so on.
  """

  payload = json.loads(request.get_data().decode('utf-8'))
  
  data = {}
  data["relayed"] = payload # just for debugging

  return json.dumps(data)


# run initial setup
initial_setup()