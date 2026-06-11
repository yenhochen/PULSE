#!/bin/sh
# Backward-compatible entry point — delegates to build.sh
sh "$(dirname "$0")/build.sh" "$@"
