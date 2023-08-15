import requests
import patoolib
import os

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id, 'confirm' : 't' }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def download_data(file_id, destination, outdir):
    if not os.path.exists(outdir):
        download_file_from_google_drive(file_id, destination)
        patoolib.extract_archive(destination, outdir=outdir)
        os.remove(destination)
        
if __name__ == "__main__":
    if not os.path.exists("models"):
        os.makedirs("models")
    download_data(file_id = '1tKQkCEO-TasGVrunkNWhuaJSP9khobph', destination = 'concept_extraction_models_en.zip', outdir = os.path.join("models", "en"))
