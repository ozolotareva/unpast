FROM python:3.10.15

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for R
RUN apt-get update && \
    apt-get install -y software-properties-common libcurl4-openssl-dev libssl-dev libxml2-dev r-base && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (Poetry)
COPY pyproject.toml poetry.lock /app/
RUN pip install --upgrade pip && \
    pip install poetry

# Copy your project files into the container
COPY . /app

# Build the Python package using Poetry
RUN poetry config virtualenvs.create false && \
    poetry build

# Install the built Python package (your package 'unpast')
RUN pip install dist/unpast-*-py3-none-any.whl

# Install BiocManager and core R packages
RUN R -e "install.packages('BiocManager')"
RUN R -e "BiocManager::install(version = '3.16')"
RUN R -e "BiocManager::install('limma')"
RUN R -e "BiocManager::install('WGCNA')"

# If WGCNA is not available, install it directly via install.packages
# RUN R -e "if (!requireNamespace('WGCNA', quietly = TRUE)) install.packages('WGCNA')"

# Create a non-root user and switch to it
RUN useradd -m user
USER user

# Set the entry point for the Docker container
ENTRYPOINT ["unpast"]
CMD ["-h"]
