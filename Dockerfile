# Agenthon 2026 Track 2 submission image.  linux/amd64 by contract.
#
# Two stages: the builder carries a CA bundle so the package index is reachable from behind a
# TLS-terminating proxy (inert on a normal network, including GitHub's runners), which keeps the
# shipped image free of any build-host certificate and of the pip cache.  The final image holds
# python, three pinned libraries and one agent file — nothing else to go wrong at run time.
ARG PY=3.12-slim
FROM python:${PY} AS builder
COPY ca-bundle.crt /tmp/ca-bundle.crt
ENV PIP_CERT=/tmp/ca-bundle.crt \
    SSL_CERT_FILE=/tmp/ca-bundle.crt \
    REQUESTS_CA_BUNDLE=/tmp/ca-bundle.crt
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
 && /opt/venv/bin/pip install --no-cache-dir \
      "numpy==2.5.3" \
      "pandas==3.0.5" \
      "pyarrow==25.0.1"

FROM python:${PY}
LABEL qfbench2.interface_version="2.0"
LABEL org.opencontainers.image.title="agenthon-t2-forecaster"

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY agent.py /app/agent.py
COPY forecast /usr/local/bin/forecast
RUN chmod +x /usr/local/bin/forecast \
 && /opt/venv/bin/python -c "import numpy, pandas, pyarrow" \
 && /opt/venv/bin/python /app/agent.py --help > /dev/null

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

# No ENTRYPOINT: the harness passes the verb as the container command, resolved from PATH.
CMD ["forecast", "--help"]
