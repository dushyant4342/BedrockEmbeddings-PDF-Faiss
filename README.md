# Streamlit Web app Docker Image (Linux) - Resume Ineteraction



When starting Docker, mount the EC2 folder inside the container:



docker build -t dushyant1334/titanembedpdf2text:latest .

docker run -v /home/ec2-user/BedrockEmbeddings-PDF-Faiss/data:/app/data -p 8501:8501 -e AWS_ACCESS_KEY_ID="" -e  AWS_SECRET_ACCESS_KEY="" -e  AWS_REGION="ap-south-1" dushyant1334/titanembedpdf2text:latest

This mounts your EC2 folder inside the container as /app/data.

This ensures /home/ec2-user/BedrockEmbeddings-PDF-Faiss/data/ in EC2 is mapped inside Docker as /app/data/.

#When you run your Docker container, the faiss_index and data folders are stored inside the container's filesystem, not directly on the EC2 #instance.

#Next Steps
AWS Load Balancer for high traffic
Set up HTTPS with Nginx for security - Buy Domain from Godaddy, allow www. from security groups.
Autoscale with ECS instead of EC2 if needed
Make good UI 
how come default resume is not there once new pdf is uploaded
Save all the pdf on ec2 instance (not in DockerImage)


sudo yum install -y nginx
sudo systemctl start nginx
sudo systemctl enable nginx

Amazon Linux 2023 uses dnf instead of yum, and EPEL is included in the default repositories.
sudo dnf install -y certbot python3-certbot-nginx
certbot --version
certbot 2.6.0

Obtain and Configure SSL Certificate
Replace yourdomain.com with your actual domain and run:
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
sudo certbot --nginx -d interactivepdf.com -d www.interactivepdf.com
sudo certbot --nginx -d pdfgeniusai.com -d www.pdfgeniusai.com



docker rm $(docker ps -a -q) # remove all stopped container

docker rmi -f $(docker images -q) #Delete all docker images
