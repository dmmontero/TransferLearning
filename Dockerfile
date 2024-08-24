# Pyrhon Dockerfile version 3.11.9 
FROM python:3.11.9

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Work dir for app 
WORKDIR /appcv

# Install pip requirements
COPY ./requirements.txt /appcv/requirements.txt

# Install reequired packages
# RUN python -m pip install -r requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install "fastapi[standard]"

# Copy project in workdir folder
COPY . /appcv

# Expose the port on which the application will run
EXPOSE 8282

# add user for execute app
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /appcv

USER appuser

# Run app
CMD ["fastapi", "run", "main.py","--host", "0.0.0.0", "--port", "8282"]