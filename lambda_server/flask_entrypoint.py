import random
from flask import Flask
from flask import request
from flask import json

from src import utils

app = Flask(__name__)

#globals: to be initialized in initial_setup - a common storage variable for everything, including initiated models and current states of each of the users.
STORAGE = {
  "static": {},
  "users": {}
}
SERVER_STATUS = "Not Ready"


def initial_setup():
  """
  Initial setup for the server
    - Load all the models.
    - Load context pairs and be ready for answering the requests.
  """
  global STORAGE
  global SERVER_STATUS

  # load all the models - mod is the model object that will be used for reading, writing, etc.
  name2mod = utils.load_models()
  STORAGE["static"]["name2mod"] = name2mod

  # load all context pairs - assume a fixed order - (Human, Agent)
  ctx_pairs = utils.load_context_pairs()
  STORAGE["static"]["ctx_pairs"] = ctx_pairs

  # initialize the storage for users
  STORAGE["users"]["mod_cxt_used"] = set() # set of (model, context) pairs that have been used.
  STORAGE["users"]["user_data"] = {} # dict from random IDs to everything related to a specific user, model, cxt, and all lioness interaction.

  # mark the server as ready
  
  SERVER_STATUS = "Ready"


@app.route('/setup_new_user/', methods=['POST'])
def setup_new_user():
  """
  Setup for a new user - build the connection.
   - return a new (model, context, random id)
   - save the model, context and random id in the storage.
   - mark this triplet as used.
  """
  global STORAGE

  if SERVER_STATUS != "Ready":
    # server is not yet ready with the initial setup.
    data = {}
    data["status"] = "Error"
    data["error_description"] = "Server is not ready yet. Please wait for a few seconds and try again."
    return json.dumps(data)

  # pick 10 random digits from 0 to 9 (both included) and join them as the randomId for the user
  randomId = ''.join([str(random.randint(0, 9)) for _ in range(10)])

  # pick a new model, cxt pair that is not in mod_cxt_used
  chosen_mod_cxt = None
  for mod in STORAGE["static"]["name2mod"]:
    for cxt in STORAGE["static"]["ctx_pairs"]:
      if (mod, cxt) not in STORAGE["users"]["mod_cxt_used"]:
        chosen_mod_cxt = (mod, cxt)
        break
    if chosen_mod_cxt:
      break
  
  # if no new model, cxt pair is available, return an error
  if not chosen_mod_cxt:
    # choose any model and a cxt pair at random
    chosen_mod_cxt = (random.choice(list(STORAGE["static"]["name2mod"].keys())), random.choice(STORAGE["static"]["ctx_pairs"]))
  
  # update the storage appropriately
  STORAGE["users"]["mod_cxt_used"].add(chosen_mod_cxt)
  STORAGE["users"]["user_data"][randomId] = {
    "model": chosen_mod_cxt[0],
    "cxt": chosen_mod_cxt[1],
    "lioness": {}, # start an account for this user in the storage
  }

  # output
  data = {}
  data["status"] = "Success"
  data["randomId"] = randomId
  data["model"] = chosen_mod_cxt[0]
  data["cxt"] = chosen_mod_cxt[1]
  return json.dumps(data)


@app.route('/model_resp/', methods=['POST'])
def model_resp():
  """
  Get model response - interface for all the cases possible. The method in utils processes the request, and returns the response object (that is sent out), and storage obj (that is used to update the lioness storage for the user).
  """
  global STORAGE
  if SERVER_STATUS != "Ready":
    # server is not yet ready with the initial setup.
    data = {}
    data["status"] = "Error"
    data["error_description"] = "Server is not ready yet. Please wait for a few seconds and try again."
    return json.dumps(data)

  payload = json.loads(request.get_data().decode('utf-8'))

  # check if the randomId is valid
  if payload["randomId"] not in STORAGE["users"]["user_data"]:
    # invalid randomId
    data = {}
    data["status"] = "Error"
    data["error_description"] = "Invalid randomId. Please check the randomId and try again."
    return json.dumps(data)
  else:
    # check if the model and cxt are valid
    if payload["model"] != STORAGE["users"]["user_data"][payload["randomId"]]["model"] or payload["cxt"] != STORAGE["users"]["user_data"][payload["randomId"]]["cxt"]:
      # invalid model or cxt
      data = {}
      data["status"] = "Error"
      data["error_description"] = "Invalid model or cxt. Please check the model and cxt and try again."
      return json.dumps(data)
  
  # get the response from the model
  resp_obj, store_obj = utils.get_model_resp(payload, STORAGE["users"]["user_data"][payload["randomId"]]["lioness"])

  #update internal storage using store_obj
  STORAGE["users"]["user_data"][payload["randomId"]]["lioness"] = store_obj

  # output
  data = {}
  data["status"] = "Success"
  data["response"] = resp_obj
  return json.dumps(data)


@app.route('/reset/', methods=['POST'])
def reset():
  """
  Reset the internal states, as if no user has been connected to the server yet, after the server has been started. You can still keep the models loaded.
  This is useful for debugging.
  """
  global STORAGE

  if SERVER_STATUS != "Ready":
    # server is not yet ready with the initial setup.
    data = {}
    data["status"] = "Error"
    data["error_description"] = "Server is not ready yet. Please wait for a few seconds and try again."
    return json.dumps(data)

  # re-initialize the storage for users
  
  STORAGE["users"]["mod_cxt_used"] = set()
  STORAGE["users"]["user_data"] = {}
  
  # output
  data = {}
  data["status"] = "Success"
  return json.dumps(data)


@app.route('/report_stats/', methods=['POST'])
def report_stats():
  """
  Report basic stats about the current state of the server.
    - # of models, contexts, etc.
    - how many users have connected so far.
    - how many users are remaining, so on.
  This is useful for debugging.
  """

  if SERVER_STATUS != "Ready":
    # initial setup has not been done yet.
    data = {}
    data["status"] = "Error"
    data["error_description"] = "Server is not ready yet. Please wait for a few seconds and try again."
    return json.dumps(data)
  
  # compute all useful stats from STORAGE
  data = {}
  data["status"] = "Success"
  data["server_status"] = SERVER_STATUS
  data["num_models"] = len(STORAGE["static"]["name2mod"])
  data["model_names"] = list(STORAGE["static"]["name2mod"].keys())
  data["num_contexts"] = len(STORAGE["static"]["ctx_pairs"])
  data["num_mod_cxt_used"] = f'{len(STORAGE["users"]["mod_cxt_used"])} / {len(STORAGE["static"]["name2mod"]) * len(STORAGE["static"]["ctx_pairs"])}'
  data["num_users_served"] = len(STORAGE["users"]["user_data"])
  
  return json.dumps(data)


# run initial setup
initial_setup()