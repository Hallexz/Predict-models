FROM python:3.9

COPY requirements.txt setup.py /workdir/
COPY src/ /workdir/src/

WORKDIR /workdir

RUN pip install -U -e .
RUN pip install uvicorn 

# Run the application
CMD ["uvicorn", "src.web_app:app", "--host", "0.0.0.0", "--port", "80"]

