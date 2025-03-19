import os

for filename in os.listdir():
    if filename.endswith(".txt"):
        with open(filename, "r") as file:
            content = file.read().replace("\n", "")

        with open(filename, "w") as file:
            file.write(content)