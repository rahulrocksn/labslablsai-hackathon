# Deploying Mirror to Vultr

This guide outlines the steps to deploy the Mirror application to a Vultr Cloud Compute instance.

## Prerequisites

1.  A [Vultr Account](https://www.vultr.com/).
2.  An SSH Key pair (optional but recommended for security).

## Step 1: Provision a Server

1.  Log in to your Vultr Dashboard.
2.  Click **"Deploy New Server"**.
3.  **Choose Server**: Cloud Compute (vCPU) is sufficient.
4.  **CPU & Storage Technology**: AMD High Performance or Intel High Performance.
5.  **Server Location**: Choose a region close to you (e.g., Silicon Valley, Frankfurt).
6.  **Server Image**: Choose **Ubuntu 22.04 LTS x64**.
7.  **Server Size**: A standard plan with at least **2GB RAM** (approx $10/mo) is recommended for computer vision tasks.
8.  **Add SSH Keys**: Select your SSH key if available, or just use the root password provided after deployment.
9.  Click **"Deploy Now"**.

Wait for the server to finish installing. Note down the **IP Address** and **Root Password**.

## Step 2: Connect to the Server

Open your terminal (PowerShell on Windows, Terminal on Mac/Linux) and SSH into your new server:

```bash
ssh root@<YOUR_SERVER_IP>
```

*Replace `<YOUR_SERVER_IP>` with the actual IP address from the Vultr dashboard.*

## Step 3: Deployment

Once logged in, perform the following steps:

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/rahulrocksn/labslablsai-hackathon.git
    cd labslablsai-hackathon
    ```

2.  **Make the Setup Script Executable**:
    ```bash
    chmod +x setup_vultr.sh
    ```

3.  **Run the Setup Script**:
    This script will install Python, OpenCV system dependencies, and start the app.
    ```bash
    ./setup_vultr.sh
    ```

## Step 4: Access the Application

Once the script finishes, copy your server's IP address and paste it into your browser with port `8501`:

```
http://<YOUR_SERVER_IP>:8501
```

You should see the Mirror application running!

## Troubleshooting

-   **App not loading?** Check if the firewall is blocking port 8501. The setup script attempts to open it, but if you have a Vultr Firewall Group active, you may need to open port 8501 in the Vultr Dashboard settings for your instance.
-   **Logs**: Check the application logs by running:
    ```bash
    tail -f streamlit.log
    ```
-   **Stopping the App**:
    Find the process ID and kill it:
    ```bash
    pkill streamlit
    ```
