# Demo 1: Real-Time Web Demo
See [our YouTube video](https://www.youtube.com/watch?v=LZJOTRkjZA4)
of using this in a real-time web application
for face recognition.
The source is available in [demos/web](https://github.com/cmusatyalab/openface/blob/master/demos/web).


<a href='https://www.youtube.com/watch?v=LZJOTRkjZA4'><img src='../../images/youtube-web.gif'></img></a>

This demo does the full face recognition pipeline on every frame.
In practice, object tracking
[like dlib's](http://blog.dlib.net/2015/02/dlib-1813-released.html)
should be used once the face recognizer has predicted a face.

To run on your system, after following the setup directions
below, install the requirements for the web demo with
`./install-deps.sh` and `sudo pip install -r requirements.txt`
from the `demos/web` directory.
The application is split into a processing server and static
web pages that communicate via web sockets.
Start the server with `./demos/web/server.py` and
serve the static website with `python2 -m SimpleHTTPServer 8000`
from the `/demos/web` directory.
You should now be able to access the demo from your browser
at `http://localhost:8000`.
The saved faces are only available for the browser session.
