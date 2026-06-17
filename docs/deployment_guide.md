# Production Deployment Guide - AI Revenue Intelligence

This guide outlines the steps to deploy the AI Revenue Intelligence platform to various hosting environments.

---

## 1. Local Development Setup

### Prerequisites
- Python 3.8 to 3.11 installed
- Git installed

### Setup Commands
```bash
# Clone the repository
git clone https://github.com/yourusername/AI-Revenue-Intelligence.git
cd AI-Revenue-Intelligence

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Pre-train models
python src/train.py --data ./data --pickle-dir ./pickle

# Run the Streamlit application
streamlit run streamlit_app.py
```

---

## 2. Docker Containerization

We provide a production-ready `Dockerfile` and `docker-compose.yml` structure.

### Create Dockerfile
Write a file named `Dockerfile` in the root:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Run model training during build to ensure pickle exists
RUN python src/train.py --data ./data --pickle-dir ./pickle

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Build and Run Docker Image
```bash
# Build the image
docker build -t ai-revenue-intelligence .

# Run the container
docker run -p 8501:8501 --env GEMINI_API_KEY="your_api_key_here" ai-revenue-intelligence
```

---

## 3. Deploys to Streamlit Community Cloud (Recommended for Hackathons)

Streamlit Community Cloud is the fastest way to deploy for judge evaluations.

1. **Push your code to GitHub**: Ensure `pickle/model.pkl` is committed to the repository (git-lfs might be required if the file is over 100MB; our packaged pickle size is small, usually ~15-20MB, so direct commits are fine).
2. **Log in to Streamlit Cloud**: Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. **Deploy App**:
   - Choose your repository.
   - Choose the branch (e.g., `main`).
   - Set the main file path to `streamlit_app.py`.
4. **Configure Secrets**:
   Click **"Advanced Settings"** before launching, and insert secrets in TOML format:
   ```toml
   GEMINI_API_KEY = "your_actual_gemini_api_key"
   OPENAI_API_KEY = "your_actual_openai_api_key"
   ```
5. Click **"Deploy"**.

---

## 4. Deploys to Hugging Face Spaces

Hugging Face Spaces provides excellent free hosting with support for Docker and Streamlit templates.

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2. Select **"Streamlit"** as the SDK (or choose **"Docker"** and use the Dockerfile template).
3. If using Streamlit SDK, upload files directly or link your Git remote.
4. Go to **"Settings"** -> **"Variables and secrets"** to add `GEMINI_API_KEY` or `OPENAI_API_KEY`.
5. The Space will build and run automatically.

---

## 5. Enterprise Hosting on AWS (EC2/ECS)

For a secure, enterprise-grade deployment:

1. **Launch an EC2 Instance**: Use an Ubuntu Server AMI (t3.medium or larger recommended).
2. **Set up Nginx**: Use Nginx as a reverse proxy to manage SSL connections and map port 80/443 to the Streamlit port 8501.
   ```nginx
   server {
       listen 80;
       server_name yourdomain.com;

       location / {
           proxy_pass http://localhost:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_real_ip_header X-Real-IP;
       }
   }
   ```
3. **Configure SSL**: Run Certbot (Let's Encrypt) to set up free, automated HTTPS certificates.
   ```bash
   sudo apt-get install certbot python3-certbot-nginx
   sudo certbot --nginx -d yourdomain.com
   ```
4. **Process Management**: Run the Streamlit process in the background using `pm2` or a systemd service to keep the application running continuously after SSH exit:
   ```bash
   pm2 start "streamlit run streamlit_app.py" --name revenue-intelligence
   ```
