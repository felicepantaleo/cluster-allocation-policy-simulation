#!/bin/bash
# Log in to monit-grafana.cern.ch via CERN SSO: Kerberos (SPNEGO) + OTP.
# Usage: monit_login.sh <otp-code> <cookie-jar>
# Leaves a Grafana session in the cookie jar; verify with /api/user.
set -euo pipefail
OTP=$1
J=$2
rm -f "$J"

PAGE=$(curl -sSL -c "$J" -b "$J" "https://monit-grafana.cern.ch/login/generic_oauth")
KRB=$(echo "$PAGE" | grep -oE 'href="/auth/realms/cern/broker/kerberos/login[^"]*"' \
      | head -1 | sed 's/href="//;s/"$//;s/&amp;/\&/g')
[ -n "$KRB" ] || { echo "no kerberos link found (no ticket? run kinit)"; exit 1; }

P2=$(curl -sSL --negotiate -u : -c "$J" -b "$J" "https://auth.cern.ch$KRB")
ACTION=$(echo "$P2" | grep -oE '<form id="kc-otp-login-form"[^>]*action="[^"]*"' \
         | sed 's/.*action="//;s/"$//;s/&amp;/\&/g')
[ -n "$ACTION" ] || { echo "no OTP form (flow changed?)"; exit 1; }

curl -sSL -c "$J" -b "$J" -o /dev/null \
     --data-urlencode "otp=$OTP" --data-urlencode "login=Sign In" "$ACTION"
USER_JSON=$(curl -sS -c "$J" -b "$J" "https://monit-grafana.cern.ch/api/user")
echo "$USER_JSON" | grep -q '"login"' \
    && { echo "login OK: $USER_JSON"; exit 0; } \
    || { echo "login FAILED: $USER_JSON"; exit 1; }
