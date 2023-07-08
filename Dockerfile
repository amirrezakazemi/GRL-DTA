FROM python:3.8-slim

# Set the working directory
WORKDIR /app


COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pip install pipenv

RUN pipenv install --system --deploy

COPY ["app.py", "data_prep.py", "./"]

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "app:application"]

