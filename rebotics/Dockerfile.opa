ARG OPA_VERSION=0.45.0-rootless
FROM openpolicyagent/opa:${OPA_VERSION}
COPY --chown=${USER} ./cvat/apps/iam/rules /rules
CMD ["run", "--server", "--addr", ":8181", "--log-level=error", "--set=decision_logs.console=true", "/rules"]
