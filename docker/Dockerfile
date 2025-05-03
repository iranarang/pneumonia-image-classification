FROM python:3.11

# Install dependencies
RUN pip install tensorflow==2.15
RUN pip install Flask==3.0
RUN pip install Pillow==10.0  # For image processing

# Copy model and api.py
COPY models /models
COPY api.py /api.py

# Set default command
CMD ["python", "api.py"]
