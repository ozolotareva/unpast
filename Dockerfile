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

# RUN R CMD INSTALL /app/r_packages/GenomeInfoDbData_1.2.11.tar.gz
# RUN R -e "BiocManager::install(c('BiocGenerics', 'GenomeInfoDb', 'IRanges', 'XVector', 'Biostrings', 'crayon'))"
# COPY r_packages /app/r_packages
# RUN R -e "BiocManager::install('KEGGREST')"
# RUN R -e "if (!requireNamespace('KEGGREST', quietly = TRUE)) { BiocManager::install('KEGGREST') }"
# RUN R -e "BiocManager::install(c('Biobase', 'DBI', 'RSQLite', 'AnnotationDbi'))"
# # RUN R CMD INSTALL /app/r_packages/AnnotationDbi_1.64.1.tar.gz
# RUN R CMD INSTALL /app/r_packages/GO.db_3.18.0.tar.gz

# Install WGCNA and limma R packages
# RUN R -e "BiocManager::install(c('WGCNA', 'limma'))"

# Create a non-root user and switch to it
RUN useradd -m user
USER user

# Set the entry point for the Docker container
ENTRYPOINT ["unpast"]
CMD ["-h"]
