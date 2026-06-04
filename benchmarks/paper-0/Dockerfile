FROM condaforge/miniforge3:latest

WORKDIR /work/pheat
COPY . /work/pheat
RUN mamba env update -n base -f benchmarks/paper-0/environment-paper.yml && \
    python -m pip install -e ".[all]"

CMD ["make", "-C", "benchmarks/paper-0", "reproduce-demo"]
