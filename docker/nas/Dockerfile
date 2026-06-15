# NAS 도커 서버 — 순수 stdlib(의존성 없음)
FROM python:3.12-slim
WORKDIR /app
COPY nas_server.py /app/nas_server.py
ENV PORT=8080 DATA_DIR=/data SEED_SHARDS=8 SAMPLES_PER_SHARD=256
VOLUME ["/data"]
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8080/health',timeout=2).status==200 else 1)"
CMD ["python3", "nas_server.py"]
