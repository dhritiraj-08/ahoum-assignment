# Dockerfile
# ----------
# Containerizes the pipeline (audit / embed / score / benchmark / app CLI
# entry points). Does NOT containerize Ollama -- Ollama must run on the
# HOST machine (this project needs a GPU for reasonable local-model
# performance, and giving a container GPU passthrough is its own can of
# worms this project doesn't need to open). The container reaches host
# Ollama via the OLLAMA_HOST env var set in docker-compose.yml
# (http://host.docker.internal:11434) -- see docs/README.md "Docker"
# section for the full explanation and docker-compose.yml's extra_hosts
# entry that makes host.docker.internal resolve on Linux.
#
# Groq is unaffected by any of this -- it's a cloud API call either way,
# containerized or not, as long as GROQ_API_KEY is set.

FROM python:3.13-slim

WORKDIR /app

# Real-time (unbuffered) stdout/stderr so `rich` console output and print()
# statements show up immediately in `docker compose logs` / `docker run`
# instead of being buffered until the process exits.
ENV PYTHONUNBUFFERED=1

# Copy requirements first, separately from the rest of the source, so
# Docker's layer cache only re-runs `pip install` when requirements.txt
# actually changes -- editing src/ files won't invalidate this layer.
COPY requirements.txt .

# Install CPU-only torch FIRST, from PyTorch's dedicated CPU wheel index,
# before requirements.txt pulls it in as a sentence-transformers dependency.
# Without this, pip resolves torch's default (CUDA-enabled) build on Linux,
# dragging in the full NVIDIA CUDA toolkit (nvidia-cublas, cudnn, cufft,
# nccl, triton, ~7GB+ of libraries) that this container will never use --
# this container has no GPU, and Ollama (the only thing that needs one)
# runs on the host, not in here (see docs/README.md "Docker" section).
# sentence-transformers only needs CPU inference here (embedding ~316
# short facet strings against a conversation, not a training workload).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Now copy everything else.
COPY . .

# Default: print CLI help. Override at `docker run`/`docker compose run`
# time with the actual command you want, e.g.:
#   docker compose run --rm facet-pipeline python main.py --audit
#   docker compose run --rm facet-pipeline python main.py --score "..."
CMD ["python", "main.py", "--help"]
