# Minimal, reproducible runner built on the Jupyter docker stack
FROM jupyter/r-notebook:latest

# Install tzdata as root so localizations work
USER root
RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*

# Switch back to the notebook user
USER ${NB_UID}
WORKDIR /home/jovyan/app

# Install Python requirements
COPY --chown=${NB_UID}:${NB_GID} requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project so Make targets have everything they need
COPY --chown=${NB_UID}:${NB_GID} . .

# Ensure writable directories exist for artifacts/data
RUN mkdir -p artifacts data weather_by_load_area

# Start the Jupyter notebook server showing the Hello World notebook by default
#CMD ["start-notebook.sh", "--NotebookApp.default_url=/lab/tree/app/hello_world.ipynb"]
CMD ["/bin/bash"]
