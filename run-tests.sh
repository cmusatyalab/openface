#!/bin/bash

set -e

cd $(dirname $0)

nosetests3 -v
