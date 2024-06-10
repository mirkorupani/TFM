# Use a Miniconda3 base image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

# Copy the environment file and requirements.txt
COPY environment.yml .
COPY requirements.txt .

# Create the environment with conda
RUN conda env create -f environment.yml

# Activate the environment and install pip packages
RUN echo "source activate your_environment_name" > ~/.bashrc
RUN /bin/bash -c "source ~/.bashrc && pip install -r requirements.txt"

# Copy the application code
COPY . .

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]