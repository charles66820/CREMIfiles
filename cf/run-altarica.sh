#!/usr/bin/env bash

IMAGE="base"
TAG=latest

usage() {
    cat <<EOF
USAGE: $0 [options]
where options are:
  -b | --base  run base image (mainly for debug purposes)
  -a | --arc  run arc model-checker
  -g | --gui  run altarica-studio GUI
  -h | --help  print this help message
EOF
}

while test $# -ne 0; do
    case "$1" in
	"-b" | "--base") IMAGE="base" ;;
	"-a" | "--arc") IMAGE="arc" ;;
	"-g" | "--gui") IMAGE="altarica-studio" ;;
	"-h" | "--help") usage; exit 0 ;;
	*)
	    (echo "bad argument '$1'"; usage ) 1>&2
	    exit 1
	    ;;
    esac
    shift
done

set -eu

DOCKER_FLAGS="--net=host"
DOCKER_FLAGS+=' -v "/tmp/.X11-unix:/tmp/.X11-unix"'
DOCKER_FLAGS+=' -v "${PWD}:/home/formaldesigner"'

case $(uname) in
    Linux*)
	DOCKER_FLAGS+=' -e DISPLAY'
	DOCKER_FLAGS+=' -e XAUTHORITY=/.Xauthority'
	DOCKER_FLAGS+=' -v "${XAUTHORITY}:/.Xauthority"'
	restore_xhost=false
	;;
    Darwin*)
	for interface in $(ifconfig -l); do
            IP="$(ipconfig getifaddr ${interface} || true)"
            if test ! -z "${IP}"; then
		break;
            fi
	done
	if test -z "${IP}"; then
	    echo 1>&2 "cannot determine IP address to connect to X server"
	    exit 1
	fi
	restore_xhost=true
	if xhost | grep -q ${IP}; then
	    restore_xhost=false
	fi
	xhost +${IP}
	DOCKER_FLAGS+=" -e DISPLAY=${IP}:0"
	if test -f "${HOME}/.Xauthority"; then
	    DOCKER_FLAGS+=' -e "XAUTHORITY=/.Xauthority"'
	    DOCKER_FLAGS+=' -v "${HOME}/.Xauthority:/.Xauthority"'
	fi
	;;
esac

eval docker run ${DOCKER_FLAGS} --rm --name alatarica-${IMAGE} -ti altarica/${IMAGE}:${TAG}
${restore_xhost} && xhost -${IP}
