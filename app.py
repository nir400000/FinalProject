from flask import Flask


app = Flask(__name__, static_folder='media', static_url_path='/media')
