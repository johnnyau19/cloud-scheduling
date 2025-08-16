FROM python:3.11.9-slim
WORKDIR /src

# Install the application dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the source code
COPY . .
EXPOSE 5000

CMD ["bash"]