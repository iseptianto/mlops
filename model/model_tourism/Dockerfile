FROM python3.9-slim
WORKDIR app
COPY . .
RUN pip install -r requirements.txt
CMD [uvicorn, appapp, --host, 0.0.0.0, --port, 8000]
