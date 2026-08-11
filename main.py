# import os
# from huggingface_hub import constants

# print("ENV:")
# print("HF_HOME =", os.environ.get("HF_HOME"))
# print("HUGGINGFACE_HUB_CACHE =", os.environ.get("HUGGINGFACE_HUB_CACHE"))

# print("\nACTUAL CACHE:")
# print("HF_HUB_CACHE =", constants.HF_HUB_CACHE)


import os

# os.environ["HF_HOME"] = "/your/path"
# os.environ["HUGGINGFACE_HUB_CACHE"] = "/your/path"

from huggingface_hub import constants

print(constants.HF_HUB_CACHE)