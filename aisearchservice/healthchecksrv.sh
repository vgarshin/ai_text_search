#!/bin/bash

TOKEN="$(cat aisearchservice/jwt_token)"

curl --fail -s \
  -H "Content-type: application/json" \
  -H "Authorization: ${TOKEN}" \
  http://localhost:50000/creds || exit 1
