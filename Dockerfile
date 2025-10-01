ARG PYTHON_VERSION=3.10.15
FROM python:${PYTHON_VERSION}

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for R
RUN apt-get update && \
    apt-get install -y libcurl4-openssl-dev libssl-dev libxml2-dev r-base && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (Poetry)
COPY pyproject.toml poetry.lock /app/
RUN pip install --upgrade pip && \
    pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root

# Install BiocManager and core R packages
RUN R -e "install.packages('BiocManager')"
RUN R -e "BiocManager::install(c('limma', 'WGCNA'), Ncpus = 4)"

# Verify WGCNA installation
RUN R -e "if (!requireNamespace('WGCNA', quietly = TRUE)) { stop('WGCNA not installed') }"

# Copy project files into the container
COPY . /app

# Build the Python package using Poetry
RUN poetry build

# Install the built unpast
RUN pip install dist/unpast-*-py3-none-any.whl

# Create a non-root user and switch to it
RUN useradd -m user
USER user

# Set the entry point for the Docker container
ENTRYPOINT ["unpast"]
CMD ["-h"]
