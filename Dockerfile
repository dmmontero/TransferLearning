# Pyrhon Dockerfile version 3.11.9 
FROM python:3.11.9

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Wor dir for app 
WORKDIR /code

# Install pip requirements
COPY ./requirements.txt /code/requirements.txt

# Install reequired packages
# RUN python -m pip install -r requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN pip install "fastapi[standard]"
# Copy project in workdir folder
COPY . /code/app

# Expose the port on which the application will run
EXPOSE 8282

# Run app
CMD ["fastapi", "run", "app/main.py","--host", "0.0.0.0", "--port", "8282"]