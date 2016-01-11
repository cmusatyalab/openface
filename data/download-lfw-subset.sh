#!/bin/bash
# Download data.

cd "$(dirname "$0")"

die() {
  echo >&2 $*
  exit 1
}

checkCmd() {
  command -v $1 >/dev/null 2>&1 \
    || die "'$1' command not found. Please install from your package manager."
}

checkCmd wget
checkCmd tar

printf "\n\n====================================================\n"
printf "Downloading lfw-a (subset of people with name starting with A)"
printf "====================================================\n\n"
rm -rf lfw-subset
mkdir -p lfw-subset
cd lfw-subset
wget http://vis-www.cs.umass.edu/lfw/lfw-a.tgz
[ $? -eq 0 ] || ( rm lfw-a.tgz && die "+ lfw-a: Error in wget." )

printf "\n\n====================================================\n"
printf "Verifying checksums.\n"
printf "====================================================\n\n"

md5str() {
  local FNAME=$1
  case $(uname) in
    "Linux")
      echo $(md5sum "$FNAME" | cut -d ' ' -f 1)
      ;;
    "Darwin")
      echo $(md5 -q "$FNAME")
      ;;
  esac
}

checkmd5() {
  local FNAME=$1
  local EXPECTED=$2
  local ACTUAL=$(md5str "$FNAME")
  if [ $EXPECTED == $ACTUAL ]; then
    printf "+ $FNAME: successfully checked\n"
  else
    printf "+ ERROR! $FNAME md5sum did not match.\n"
    printf "  + Expected: $EXPECTED\n"
    printf "  + Actual: $ACTUAL\n"
    printf "  + Please manually delete this file and try re-running this script.\n"
    return -1
  fi
  printf "\n"
}

set -e

checkmd5 \
  lfw-a.tgz \
  678b1f67c300002fedafbb0705d22e8d

tar xf lfw-a.tgz
mkdir raw
mv lfw/{Ann_Veneman,Adrien_Brody,Anna_Kournikova} raw
rm -rf lfw

rm lfw-a.tgz
