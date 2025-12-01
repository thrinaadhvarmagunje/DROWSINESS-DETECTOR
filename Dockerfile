FROM python:3.10-slim

WORKDIR /code

COPY requirements.txt /code/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /code

EXPOSE 7860

CMD ["python", "app.py"]
