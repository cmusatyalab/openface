## Demo 1: Real-Time Web Demo
Released by [Brandon Amos](http://bamos.github.io) on 2015-10-13.

---

See [our YouTube video](https://www.youtube.com/watch?v=LZJOTRkjZA4)
of using this in a real-time web application
for face recognition.
The source is available in
[demos/web](https://github.com/cmusatyalab/openface/blob/master/demos/web).
The browser portions have been tested on Google Chrome 46 in OSX.

<a href='https://www.youtube.com/watch?v=LZJOTRkjZA4'><img src='https://raw.githubusercontent.com/cmusatyalab/openface/master/images/youtube-web.gif'></img></a>

This demo does the full face recognition pipeline on every frame.
In practice, object tracking
[like dlib's](http://blog.dlib.net/2015/02/dlib-1813-released.html)
should be used once the face recognizer has predicted a face.

In the edge case when a single person is trained,
the classifier has no knowledge of other people and
labels anybody with the name of the trained person.

The web demo does not predict unknown users and the saved
faces are only available for the browser session.
If you're interested in predicting unknown people,
one idea is to use a probabilistic classifier to predict
confidence scores and then call the prediction unknown
if the confidence is too low.
See the [classification demo](http://cmusatyalab.github.io/openface/demo-3-classifier/)
for an example of using a probabilistic classifier.

See [cowbjt's unofficial fork](https://github.com/cowbjt/openface/tree/demo-web-stand-alone)
for a version of the demo that saves trained faces in a SQLite
database.

---

## Setup and Running

To run on your system, first follow the
[Setup Guide](setup.md) and make sure you can
run a simpler demo, like the [comparison demo](demo-2-comparison.md).

If you experience issues with the web demo,
please post to
[our mailing list](https://groups.google.com/forum/#!forum/cmu-openface)
and include the the WebSocket log contents from
`/tmp/openface.websocket.log` if available.

### Warning when running remotely
Trying to connect to a remote or Docker version of OpenFace in
the latest version of Chrome will result in the following error:

> getUserMedia() no longer works on insecure origins. To use this
> feature, you should consider switching your application to a secure
> origin, such as HTTPS. See https://goo.gl/rStTGz for more details.

They suggest three workarounds:

1. localhost is treated as a secure origin over HTTP, so if you're
    able to run your server from localhost, you should be able to test
    the feature on that server.

2. You can run chrome with the
    --unsafely-treat-insecure-origin-as-secure="example.com" flag
    (replacing "example.com" with the origin you actually want to test),
    which will treat that origin as secure for this session. Note that
    you also need to include the --user-data-dir=/test/only/profile/dir
    to create a fresh testing profile for the flag to work.

3. Use secure protocols

---

\#2 is requires starting Chrome with a non-standard flag.

If you don't want to start Chrome with a non-standard flag,
the following commands use [ncat](https://nmap.org/ncat/) to
route all OpenFace traffic through localhost to a remote server or
Docker container so that the demo can be accessed in Chrome
at `http://localhost:8000`.
Replace `SERVER_IP` with the IP address of your server.

```
export SERVER_IP=192.168.99.100
ncat --sh-exec "ncat $SERVER_IP 8000" -l 8000 --keep-open &
ncat --sh-exec "ncat $SERVER_IP 9000" -l 9000 --keep-open &
```

We are also interested in help running this demo with secure protocols
in [Issue #75](https://github.com/cmusatyalab/openface/issues/75)
so the demo works on a remote server or Docker without these workarounds.

### With Docker

Start the HTTP and WebSocket servers on ports 8000 and 9000 in the
Docker container with:

```
docker run -p 9000:9000 -p 8000:8000 -t -i bamos/openface /bin/bash -l -c '/root/openface/demos/web/start-servers.sh'
```

Then find the IP address of the container and access the demo
in your browser at `http://docker-ip:8000`.

### Manual Setup
After following the OpenFace setup guide and successfully running the
comparison demo, install the requirements for the web demo with
`sudo pip install -r requirements.txt`
from the `demos/web` directory.

Start the HTTP and WebSocket servers on ports 8000 and 9000, respectively,
with `./demos/web/start-servers.sh`.
If you wish to use other ports,
pass them as `./demos/web/start-servers.sh HTTP_PORT WEBSOCKET_PORT`.

You should now also be able to access the demo from your browser
at `http://localhost:8000` if running locally or
`http://your-server:8000` if running on a server.
