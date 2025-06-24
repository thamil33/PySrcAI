
# @title Imports

from json import load
import numpy as np
from IPython import display

import sentence_transformers

# Use it like any other Concordia language model
from concordia.language_model.openrouter_model import OpenRouterLanguageModel

from concordia.language_model import no_language_model

from concordia.prefabs.simulation import generic as simulation

import concordia.prefabs.entity as entity_prefabs
import concordia.prefabs.game_master as game_master_prefabs

from concordia.typing import prefab as prefab_lib
from concordia.utils import helper_functions

from dotenv import load_dotenv
load_dotenv()

import os
import time
from concordia.language_model.openrouter_model import OpenRouterLanguageModel
from concordia.language_model.retry_wrapper import RetryLanguageModel
import openai  # If using OpenAI's error classes; adjust if using another provider

_API_KEY = os.environ.get('OPENROUTER_API_KEY')
MODEL_NAME = os.environ.get('MODEL_NAME', 'mistralai/mistral-small-3.1-24b-instruct:free')

if not _API_KEY:
    raise ValueError('OPENROUTER_API_KEY is required.')

# To debug without spending money on API calls, set DISABLE_LANGUAGE_MODEL=True
DISABLE_LANGUAGE_MODEL = False

# Enable verbose logging to see which entity is making each LLM call
VERBOSE_LOGGING = True

if not DISABLE_LANGUAGE_MODEL:
    base_model = OpenRouterLanguageModel(
        api_key=_API_KEY,
        model_name=MODEL_NAME
    )
    # Enable verbose logging if supported
    if hasattr(base_model, "verbose_logging"):
        base_model.verbose_logging = True

    # Define a custom retry wrapper to handle 429 errors with a 61s wait
    class CustomRetryLanguageModel(RetryLanguageModel):
        def __init__(self, model):
            super().__init__(
                model,
                retry_on_exceptions=(Exception,),
                retry_tries=10,
                retry_delay=61,  # Wait 61 seconds on rate limit
            )

        def sample_text(self, *args, **kwargs):
            try:
                return super().sample_text(*args, **kwargs)
            except openai.error.RateLimitError as e:
                print("Rate limit hit. Waiting 61 seconds before retrying...")
                time.sleep(61)
                return super().sample_text(*args, **kwargs)

        def sample_choice(self, *args, **kwargs):
            try:
                return super().sample_choice(*args, **kwargs)
            except openai.error.RateLimitError as e:
                print("Rate limit hit. Waiting 61 seconds before retrying...")
                time.sleep(61)
                return super().sample_choice(*args, **kwargs)

    model = CustomRetryLanguageModel(base_model)
else:
    from concordia.language_model import no_language_model
    model = no_language_model.NoLanguageModel()


# In[81]:


# @title Setup sentence encoder

if DISABLE_LANGUAGE_MODEL:
  embedder = lambda _: np.ones(3)
else:
  st_model = sentence_transformers.SentenceTransformer(
      'sentence-transformers/all-mpnet-base-v2')
  embedder = lambda x: st_model.encode(x, show_progress_bar=False)


# In[83]:


test = model.sample_text(
    'Is societal and technological progress like getting a clearer picture of '
    'something true and deep?')
print(test)



# @title Load prefabs from packages to make the specific palette to use here.

prefabs = {
    **helper_functions.get_package_classes(entity_prefabs),
    **helper_functions.get_package_classes(game_master_prefabs),
}


# #@title Print menu of prefabs

# display.display(
#     display.Markdown(helper_functions.print_pretty_prefabs(prefabs)))


# @title Configure instances.

instances = [
    prefab_lib.InstanceConfig(
        prefab='basic__Entity',
        role=prefab_lib.Role.ENTITY,
        params={
            'name': 'Alice',
        },
    ),
    prefab_lib.InstanceConfig(
        prefab='basic__Entity',
        role=prefab_lib.Role.ENTITY,
        params={
            'name': 'White Rabbit',
            'goal': 'Get to the queen on time',
        },
    ),
    prefab_lib.InstanceConfig(
        prefab='generic__GameMaster',
        role=prefab_lib.Role.GAME_MASTER,
        params={
            'name': 'default rules',
            'acting_order': 'fixed',
        },
    ),
]


config = prefab_lib.Config(
    default_premise=(
        'Alice is sitting on the bank, being quite bored and tired by reading a'
        ' book with no pictures or conversations, when suddenly a white rabbit'
        ' with pink eyes ran close by her. '
    ),
    default_max_steps=2,
    prefabs=prefabs,
    instances=instances,
)


# # The simulation



# @title Initialize the simulation
runnable_simulation = simulation.Simulation(
    config=config,
    model=model,
    embedder=embedder,
)




# @title Multiple Models Example (Optional)
# This shows how to use different models for different entities

# Uncomment to try different models for different entities:

# from concordia.language_model.lmstudio_model import LMStudioLanguageModel

# # Create different models with verbose logging
# openrouter_model = OpenRouterLanguageModel(
#     api_key=_API_KEY, 
#     model_name=MODEL_NAME,
#     verbose_logging=True  # Enable detailed logging
# )

# lmstudio_model = LMStudioLanguageModel(
#     model_name="local-llama",
#     base_url="http://localhost:1234/v1",
#     verbose_logging=True  # Enable detailed logging
# )

# # To assign different models to entities, you would modify the simulation
# # initialization to pass model-specific configurations




# @title Run the simulation
results_log = runnable_simulation.play()




# @title Display the log
display.HTML(results_log)

