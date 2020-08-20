import subprocess


if __name__ == "__main__":
    output, errors = subprocess.Popen(["pip3.6", "install", "-r", "requirements.txt", "--user"],
                             stderr=subprocess.PIPE).communicate()
    if errors:
        print(errors)
    
    print("Downloading the models...")
    output, errors = subprocess.Popen(["python", "download_models.py"], stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE).communicate()
    if errors:
        print(errors)
