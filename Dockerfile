FROM python:3.5.2
ADD . /code
WORKDIR /code
RUN pip install -r requirements.txt
CMD python main.py