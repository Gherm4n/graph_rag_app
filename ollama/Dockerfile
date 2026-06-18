FROM ollama/ollama:latest

RUN nohup bash -c "ollama serve &" && \
    sleep 5 && \
    ollama pull nomic-embed-text && \
    pkill ollama