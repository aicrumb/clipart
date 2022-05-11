import subprocess

prompts = ["the capybara is wearing a little hat",
           "an old photograph of a dog made of twigs",
           "a road sign with the word 'Hello' on it",
           "iridescent vaporwave bubbles #digitalart",
           "the blue train is approaching on the tracks",
           "a pentagonal green clock"]

# python bake.py prompt
for prompt in prompts:
    subprocess.run(["python", "bake.py", prompt])
    print(f"Baked {prompt}")