#!/bin/bash

set -e -u

function die { echo $1; exit 42; }

HTTP_PORT=8000
WEBSOCKET_PORT=9000

case $# in
  0) ;;
  1) HTTP_PORT=$1
     ;;
  2) WEBSOCKET_PORT=$2
     ;;
  *) die "Usage: $0 <HTTP Server Port> <WebSocket Port>"
     ;;
esac

cd $(dirname $0)
trap 'kill $(jobs -p)' EXIT

cat <<EOF

Starting the HTTP server on port $HTTP_PORT
and the WebSocket server on port $WEBSOCKET_PORT.

Access the demo through the HTTP server in your browser.
If you're running on the same computer outside of Docker, use http://localhost:$HTTP_PORT
If you're running on the same computer with Docker, find the IP
address of the Docker container and use http://<docker-ip>:$HTTP_PORT.
If you're running on a remote computer, find the IP address
and use http://<remote-ip>:$HTTP_PORT.

WARNING: Chromium refuses to connect to the insecure WebSocket server
if you are running a remote or Docker deployment.
We have posted a workaround to forward traffic through localhost
using ncat at http://cmusatyalab.github.io/openface/demo-1-web/.
Track our progress on fixing this at:
https://github.com/cmusatyalab/openface/issues/75.


EOF

WEBSOCKET_LOG='/tmp/openface.websocket.log'
printf "WebSocket Server: Logging to '%s'\n\n" $WEBSOCKET_LOG

python2 -m SimpleHTTPServer $HTTP_PORT &> /dev/null &

cd ../../ # Root OpenFace directory.
./demos/web/websocket-server.py --port $WEBSOCKET_PORT 2>&1 | tee $WEBSOCKET_LOG &

wait
