# generate self-signed certs with no password for the web and socket servers
# this script requires that openssl is installed: e.g. sudo apt-get install openssl
mkdir tls
openssl genrsa -des3 -out tls/server.key 1024
openssl req -new -key tls/server.key -out tls/server.csr
cp tls/server.key tls/server.key.org
openssl rsa -in tls/server.key.org -out tls/server.key
openssl x509 -req -days 3650 -in tls/server.csr -signkey tls/server.key -out tls/server.crt
echo 'converting to pem'
cat tls/server.crt tls/server.key > tls/server.pem
echo 'cert complete'
