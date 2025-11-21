VENV            ?= .venv
PYTHON          ?= $(VENV)/bin/python
GO_DIR          ?= busquepet-go
COMPOSE         ?= docker compose
EPOCHS          ?= 10
BATCH           ?= 16
TRAIN_RESUME    ?=
EMBED_RESUME    ?=

.PHONY: help setup deps lint python-test go-test test serve-webhook serve-api serve-frontend dataset \
	train-contrastive train-lora embeddings faiss compose-up compose-down logs clean

help:
	@echo "Alvos principais:"
	@echo "  make setup              # cria o venv (.venv) e instala dependências Python"
	@echo "  make deps               # reaplica requirements no venv já existente"
	@echo "  make test               # pytest + go test"
	@echo "  make serve-webhook      # sobe FastAPI (IA)"
	@echo "  make serve-api          # sobe a API Go"
	@echo "  make serve-frontend     # serve o painel em frontend/"
	@echo "  make compose-up         # sobe Redis/NATS/API/Webhook via Docker"
	@echo "  make dataset|embeddings|faiss # etapas do pipeline de dados"
	@echo "Variáveis úteis:"
	@echo "  TRAIN_RESUME=auto make train-contrastive  # retoma do último checkpoint"
	@echo "  EMBED_RESUME=1 make embeddings            # continua extração de embeddings"

setup:
	python3 -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

deps:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m black --check src diagnostic.py
	$(PYTHON) -m flake8 src diagnostic.py

python-test:
	$(PYTHON) -m pytest -q

go-test:
	cd $(GO_DIR) && go test ./...

test: python-test go-test

serve-webhook:
	$(PYTHON) -m uvicorn src.webhook_service:app --host 0.0.0.0 --port 8001

serve-api:
	cd $(GO_DIR) && go run ./cmd/busquepet-api

serve-frontend:
	cd frontend && python3 -m http.server 5173

dataset:
	$(PYTHON) src/dataset_builder.py

train-contrastive:
	$(PYTHON) src/train_contrastive.py --epochs $(EPOCHS) --batch-size $(BATCH) $(if $(TRAIN_RESUME),--resume $(TRAIN_RESUME),)

train-lora:
	$(PYTHON) src/train_lora.py --epochs $(EPOCHS) --batch-size $(BATCH)

embeddings:
	$(PYTHON) src/embeddings.py $(if $(EMBED_RESUME),--resume,)

faiss:
	$(PYTHON) src/index_faiss.py

compose-up:
	$(COMPOSE) up --build

compose-down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

clean:
	rm -rf $(VENV) src/__pycache__ tests/__pycache__ .pytest_cache outputs
