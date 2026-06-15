# 엣지 추론 도커 서버 — numpy 상수메모리 스트리밍(torch 불필요, 엣지 친화)
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy
COPY edge_server.py /app/edge_server.py
ENV PORT=8091 EXPORT_DIR=/exports D_CAP=256 LAYER_CAP=4 VOCAB=64
VOLUME ["/exports"]
EXPOSE 8091
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8091/health',timeout=2).status==200 else 1)"
CMD ["python3", "edge_server.py"]
