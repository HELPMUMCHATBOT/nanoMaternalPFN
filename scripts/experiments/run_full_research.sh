#!/bin/sh
# Real-input E1-E12 research launch helper. No smoke data and no silent AWS charges.
set -eu
if [ "$#" -lt 1 ]; then
  echo 'Usage: sh scripts/experiments/run_full_research.sh {check|local-cuda|local-mps|aws-plan|aws-submit|aws-download|evaluate|verify|status} [extra args]' >&2
  exit 2
fi
MODE=$1
shift
case "$MODE" in
  check)
    exec python -m scripts.experiments.run_complete_study --phase check "$@" ;;
  local-cuda)
    exec python -m scripts.experiments.run_complete_study --phase all --device cuda --confirm-compute --require-controlled --quantize "$@" ;;
  local-mps)
    exec python -m scripts.experiments.run_complete_study --phase all --device mps --confirm-compute --require-controlled --quantize "$@" ;;
  aws-plan)
    exec python -m scripts.experiments.submit_aws_full_training "$@" ;;
  aws-submit)
    python -m scripts.experiments.run_complete_study --phase check
    exec python -m scripts.experiments.submit_aws_full_training --submit --confirm-billing "$@" ;;
  aws-download)
    exec python -m scripts.experiments.download_aws_full_training "$@" ;;
  evaluate)
    exec python -m scripts.experiments.run_complete_study --phase evaluate --require-controlled --quantize "$@" ;;
  verify)
    exec python -m scripts.experiments.verify_campaign "$@" ;;
  status)
    exec python -m scripts.experiments.run_complete_study --phase status "$@" ;;
  *)
    echo 'Unknown command' >&2
    exit 2 ;;
esac
