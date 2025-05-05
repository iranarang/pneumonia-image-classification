NEED TO MANUALLY DOWNLOAD .KERAS FILE BECAUSE IT IS TOO BIG

docker commands:
docker compose -f docker/docker-compose.yml up --build -d
docker compose -f docker/docker-compose.yml down 

curl commands:
curl -X POST -F "image=@/Users/iranarang/Documents/pneumonia-image-classification/data/test/pneumonia/person97_bacteria_468.jpeg" http://localhost:5000/inference
etc

