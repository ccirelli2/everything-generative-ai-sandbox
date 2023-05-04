import os
from milvus import default_server
from pymilvus import connections, utility

# Globals
DIR_DATA = '/home/oem/repositories/generative-ai-sandbox/vector-databases/data'

# Start Milvus Server
default_server.start()

with default_server:
    # Establish Connection w/ Local Host
    connections.connect(host='127.0.0.1', port=default_server.listen_port)

    # Set Default Data Directory
    #default_server.set_base_dir(os.path.join(DIR_DATA, 'milus_data'))

    # Check if the server is working
    print(utility.get_server_version())


