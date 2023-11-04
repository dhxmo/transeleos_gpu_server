FROM ubuntu:latest
LABEL authors="dhruv"

ENTRYPOINT ["top", "-b"]

FROM nginx
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/