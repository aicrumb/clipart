import subprocess
# import sys
# prompts = sys.argv[1:].split(",")

# these were generated when i had access to gpt3, i have a few thousand of them that i just random
# cycle through till i find a good one
# https://aicrumb.github.io/what%20am%20i/#
# its good inspiration for prompts
# be wary i filtered some words but i didnt catch all of them
# gpt3 at that time was NOT very pg
prompts = [
           "a rose in a bush",
           "a tulip on a rock",
           "a dog sitting on a hill"
           "a dictionary on a wooden table",
           "a mushroom",
           "the end of the world",
           "a yellow schoolbus"
           ]

# python bake.py prompt
for prompt in prompts:
    subprocess.run(["python", "bake.py", prompt])
    print(f"Baked {prompt}")