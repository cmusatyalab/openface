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

printf "HTTP Port: %s\n" $HTTP_PORT
printf "WebSocket Port: %s\n\n" $WEBSOCKET_PORT

WEBSOCKET_LOG='/tmp/openface.websocket.log'
printf "WebSocket Server: Logging to '%s'\n\n" $WEBSOCKET_LOG

python2 -m SimpleHTTPServer $HTTP_PORT &> /dev/null &

cd ../../ # Root OpenFace directory.
./demos/web/websocket-server.py --port $WEBSOCKET_PORT 2>&1 | tee $WEBSOCKET_LOG &

wait
