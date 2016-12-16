# Behaviour cloning


## Setting up the Python Server

The car will just sit there until your Python server connects to it and provides it steering angles. Here’s how you start your Python server:

1. Install Python Dependencies with Anaconda (conda install …)
- numpy
- flask-socketio
- eventlet
- pillow
- h5py

2. Install Python Dependencies with pip (pip install ...)
- keras

3. Download drive.py.
4. Run Server

`python drive.py model.json`

If you're using Docker for this project: `docker run -it --rm -p 4567:4567 -v `pwd`:/src udacity/carnd-term1-starer-kit python drive.py model.json or docker run -it --rm -p 4567:4567 -v ${pwd}:/src udacity/carnd-term1-starer-kit python drive.py model.json.` Port 4567 is used by the simulator to communicate.