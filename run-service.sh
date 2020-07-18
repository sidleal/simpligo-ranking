docker stop simpligo-ranking
docker rm simpligo-ranking
docker run -d --name simpligo-ranking -p 5000:8080 -v $PWD:/opt/simpligo-ranking --restart always simpligo-ranking:$1
