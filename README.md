BusquePet combina um pipeline de visão computacional pesado em Python com um front door escrito em Go que trata uploads, filas e notificações em tempo real. Esta versão deixa o dia a dia mais simples: basta iniciar a API Go, o webhook Python e os serviços de infraestrutura (Redis + NATS) para liberar pesquisas híbridas de imagens.

## Visão rápida
- **API Go (`busquepet-go`)**: recebe uploads, normaliza imagens, calcula pHash, armazena o status no Redis, publica tarefas no NATS e envia atualizações via WebSocket.
- **Webhook Python (`src/webhook_service.py`)**: consome o mesmo pipeline de inferência já existente (FAISS + Transformers) e devolve os matches para a API Go.
- **Armazenamento compartilhado**: o diretório `shared-storage/` recebe originais e normalizados para que o webhook leia o mesmo caminho recebido da API.

## Como executar localmente
1. **Dependências**  
   - Python 3.11 com os pacotes de `requirements.txt`.  
   - Go 1.22+.  
   - Redis e NATS em execução (ou use o `docker-compose.yml` descrito abaixo).  
   - Modelos e índices gerados previamente (`models/contrastive/best_model.pt`, `models/faiss_index/faiss_index.bin`, `data/embeddings/embedding_metadata.csv` etc.).

2. **Webhook Python**  
   ```bash
   export BUSQUEPET_MODEL_PATH=models/contrastive/best_model.pt
   export BUSQUEPET_FAISS_INDEX=models/faiss_index
   export BUSQUEPET_METADATA_PATH=data/embeddings/embedding_metadata.csv
   export BUSQUEPET_WEBHOOK_PORT=8001
   python3 -m uvicorn src.webhook_service:app --host 0.0.0.0 --port ${BUSQUEPET_WEBHOOK_PORT}
   ```

3. **API Go**  
   ```
   export BUSQUEPET_STORAGE_DIR=$PWD/shared-storage
   export BUSQUEPET_REDIS_URL=redis://localhost:6379/0
   export BUSQUEPET_NATS_URL=nats://localhost:4222
   export BUSQUEPET_PYTHON_WEBHOOK_URL=http://127.0.0.1:8001/v1/match
   cd busquepet-go && go run ./cmd/busquepet-api
   ```

4. **Fluxo**  
   - Faça upload em `POST /v1/ingest` (campo multipart `image`).  
   - Acompanhe o status em tempo real via `GET /v1/jobs/{job_id}` ou WebSocket em `/v1/ws`.  
   - O webhook recebe a notificação, roda a busca híbrida e devolve os matches que ficam disponíveis no Redis para as consultas da API.

## Execução com Docker Compose
1. Crie pastas locais: `mkdir -p shared-storage models data`. Copie os modelos/índices e dados processados para estas pastas.
2. Suba tudo com um único comando:
   ```bash
   docker compose up --build
   ```
3. Serviços expostos:
   - API Go: http://localhost:8080
   - Webhook Python: http://localhost:8001
   - Redis: localhost:6379
   - NATS: localhost:4222 (painel em 8222)

O arquivo `docker-compose.yml` já liga os serviços, monta o diretório compartilhado e injeta as variáveis exigidas. Basta preencher os dados de `models/` e `data/`.

## Contratos principais
- **Webhook Python (`POST /v1/match`)**  
  Corpo:
  ```json
  {
    "job_id": "123",
    "image_path": "/workspace/shared/normalized/<id>.jpg",
    "phash": "abcd...",
    "k": 10,
    "use_hybrid": true,
    "metadata": {
      "original_filename": "...",
      "checksum": "...",
      "received_at": "2024-11-14T17:06:00Z"
    }
  }
  ```
  Resposta:
  ```json
  {
    "job_id": "123",
    "matches": [
      {"rank": 1, "image_id": "...", "breed": "...", "score": 0.98, "distance": 0.03, "image_path": "..."}
    ],
    "duration_ms": 123
  }
  ```

- **API Go (`POST /v1/ingest`)**  
  Multipart com `image` → retorna `{ job_id, status, phash }` e dispara todo o pipeline automaticamente.

## Dicas de validação
1. Rode `go test ./...` dentro de `busquepet-go` para checar dependências.
2. Use `python3 -m py_compile src/webhook_service.py` para garantir que o webhook carrega.
3. Após subir tudo, envie um upload real e acompanhe o ciclo completo pelo WebSocket: você verá mensagens `job.status` (pending → processing → completed) seguidas de `job.result`.

## Implantação
- Garanta que Redis e NATS estejam altamente disponíveis (ou configure URLs externos via variáveis).
- Monte os diretórios de modelos e dados como volumes somente leitura para o webhook.
- Exporte `BUSQUEPET_STORAGE_DIR` para um disco que possa ser compartilhado com o serviço Python, pois o caminho enviado no webhook é absoluto.
- Monitore o consumo do worker Go: se necessário, escale múltiplas instâncias apontando para o mesmo NATS subject (`busquepet.ingest`).

Com isso você tem um fluxo natural: o time de dados continua evoluindo o backbone em Python enquanto o backend Go garante ingestão e entrega em produção.
