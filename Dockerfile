FROM python:3.8.3

WORKDIR /home/segment-brain-mri

COPY . /home/segment-brain-mri

RUN pip install -r requirements.txt
	
EXPOSE 5000

