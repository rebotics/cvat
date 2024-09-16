ARG OPA_VERSION=0.63.0
FROM openpolicyagent/opa:${OPA_VERSION}
COPY --chown=${USER} ./cvat/apps/analytics_report/rules/*.rego /rules/
COPY --chown=${USER} ./cvat/apps/engine/rules/*.rego /rules/
COPY --chown=${USER} ./cvat/apps/events/rules/*.rego /rules/
COPY --chown=${USER} ./cvat/apps/iam/rules/*.rego /rules/
COPY --chown=${USER} ./cvat/apps/lambda_manager/rules/*.rego /rules/
COPY --chown=${USER} ./cvat/apps/log_viewer/rules/*.rego /rules/
COPY --chown=${USER} ./cvat/apps/organizations/rules/*.rego /rules/
COPY --chown=${USER} ./cvat/apps/quality_control/rules/*.rego /rules/
COPY --chown=${USER} ./cvat/apps/webhooks/rules/*.rego /rules/

CMD ["run", "--server", "--addr", ":8181", "--log-level=error", "--set=decision_logs.console=true", "/rules"]
#CMD ["run", "--server", "--log-level=error", "--set=services.cvat.url=\${CVAT_URL}", "--set=bundles.cvat.service=cvat",
#     "--set=bundles.cvat.resource=/api/auth/rules", "--set=bundles.cvat.polling.min_delay_seconds=5",
#     "--set=bundles.cvat.polling.max_delay_seconds=15"]
