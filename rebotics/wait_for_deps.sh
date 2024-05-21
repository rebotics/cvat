#!/bin/sh

# Copyright (C) 2023 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

# This is a wrapper script for running backend services. It waits for services
# the backend depends on to start before executing the backend itself.

# Ideally, the check that all DB migrations have completed should also be here,
# but it's too resource-intensive to execute for every worker we might be running
# in a container. Instead, it's in backend_entrypoint.sh.

~/wait-for-it.sh "${DB_URL}" -t 0
~/wait-for-it.sh "${REDIS_URL}" -t 0

exec "$@"
