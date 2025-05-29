# ─────────────────────────────────────────────────────────────────────────────
# Stage 0 – build C++ extensions (needs compilers & dev libs)
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.9-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
# Install only deps necessary for compilation (numpy, pybind11)
RUN pip install --no-cache-dir numpy pybind11

COPY cpp_src/ cpp_src/
COPY bindings/ bindings/
COPY setup.py .

# Compile in-place so .so files sit inside package tree
RUN python setup.py build_ext --inplace

# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – production runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1 \
    C_FORCE_ROOT=1 \
    FLASK_ENV=production \
    CELERY_BROKER_URL=redis://redis:6379/0 \
    CELERY_RESULT_BACKEND=redis://redis:6379/0

# system libs for numpy / scipy FFT-pack
RUN apt-get update && \
    apt-get install -y --no-install-recommends libopenblas-base && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ncorr_app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# ── copy compiled extensions + source code from builder
COPY --from=builder /build/ncorr_app/_ext ./ncorr_app/_ext
COPY ncorr_app/ ncorr_app/
COPY webapp/ webapp/
COPY config_files/ config_files/
COPY config_files/default_settings.yaml .

# ── expose static dir for Nginx
RUN mkdir -p /usr/share/nginx/html && \
    cp -r webapp/static/ /usr/share/nginx/html/static

EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-k", "gevent", "-t", "120", "-b", "0.0.0.0:8000", "webapp.run:app"]
