#!/bin/bash

set -x -e

#bower install

rm -rf vendor
mkdir -p vendor/{css,fonts,js}

cp -r bower_components/bootstrap/dist/* vendor
cp -r bower_components/bootstrap-toggle/css/*.min.css* vendor
cp -r bower_components/bootstrap-toggle/js/*.min.js* vendor
cp -r bower_components/bootstrap3-dialog/dist/* vendor
cp -r bower_components/font-awesome/* vendor

cp bower_components/jquery/dist/* vendor/js
cp bower_components/handlebars/handlebars.min.js vendor/js
cp bower_components/underscore-min.js vendor/js

wget https://raw.githubusercontent.com/jstat/jstat/1.3.0/dist/jstat.min.js -P vendor/js
