FROM python:3.9
WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

# Generate pickle files first, then register model
CMD ["sh", "-c", "python generate_tourism_data.py && python register_model.py"]