#!/bin/bash

# setup_vultr.sh
# Script to set up the environment and run the Mirror app on a Vultr Ubuntu server.

# 1. Check for root privileges
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root or using sudo"
  exit 1
fi

# 2. Update system and install system dependencies
echo "Updating system packages..."
apt-get update
apt-get install -y python3-pip python3-venv git libgl1-mesa-glx libglib2.0-0 curl

# 3. Create a virtual environment (optional but recommended)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
source venv/bin/activate || echo "Operating outside venv (as root)"

# 4. Install Python dependencies
echo "Installing Python requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "requirements.txt not found! Please ensure you are in the project directory."
    exit 1
fi

# 5. Open firewall port for HTTP (Port 80)
echo "Configuring firewall..."
ufw allow 80/tcp

# 6. Get IP Address from .env or Detect
PUBLIC_IP=""

if [ -f .env ]; then
    echo "Reading IP from .env file..."
    # Read key=val format. We only care about VULTR_PUBLIC_IP
    PUBLIC_IP=$(grep "^VULTR_PUBLIC_IP=" .env | cut -d '=' -f2)
fi

if [ -z "$PUBLIC_IP" ]; then
    echo "Detecting Public IP Address (Auto)..."
    PUBLIC_IP=$(curl -s ifconfig.me)
fi

if [ -z "$PUBLIC_IP" ]; then
    PUBLIC_IP="localhost"
fi

# 7. Run the application on Port 80
echo ""
echo "========================================================"
echo "Setup complete! Running Streamlit app..."
echo "Access your app at: http://$PUBLIC_IP/"
echo "(Note: Running via nohup. Logs are in streamlit.log)"
echo "========================================================"
echo ""

# Run in background with nohup on Port 80
# We use pkill to stop any existing instance first
pkill -f "streamlit run app.py" || true

nohup streamlit run app.py --server.port 80 --server.address 0.0.0.0 > streamlit.log 2>&1 &

echo "App is running! Use 'tail -f streamlit.log' to see logs."
