# BusquePet - Fluxo Completo (Obsidian)

```mermaid
flowchart LR
    client(Cliente: upload/consulta)
    api[API Go /v1/ingest]
    preproc[Pré-processa + normaliza imagem]
    storage["shared-storage/ (originais + normalizados)"]
    phash[Calcula pHash]
    redis[Redis: status/jobs]
    nats[NATS: busquepet.ingest]
    worker[Worker Go]
    python[Webhook Python /v1/match]
    pipeline["Pipeline híbrido (embedding + pHash)"]
    faiss[FAISS + metadata CSV]
    results[Redis: results/matches]
    ws[WebSocket /v1/ws]
    httpGet[GET /v1/jobs/{id}]

    client -->|POST imagem| api
    api --> preproc --> storage
    preproc --> phash
    api -->|salva status pending| redis
    api -->|publica tarefa| nats
    api -->|status pending| ws

    nats --> worker
    worker -->|status processing| redis
    worker -->|notify status| ws
    worker -->|POST imagem normalizada + pHash| python

    python --> pipeline --> faiss
    pipeline -->|matches| python
    python -->|resposta JSON| worker

    worker -->|salva matches| results
    worker -->|status completed/failed| redis
    worker -->|job.result + status| ws

    client <-->|status/result| ws
    client -->|GET status/result| httpGet
    httpGet --> redis
    httpGet --> results
```

```mermaid
flowchart LR
    subgraph Ingestao
        A[Upload /v1/ingest] --> B[Valida/normaliza]
        B --> C[pHash]
        C --> D[Redis: job pending]
        D --> E[NATS fila ingest]
    end

    subgraph Worker
        E --> F[Worker Go]
        F --> G[Redis: processing]
        F --> H[/Webhook Python/]
    end

    subgraph Match
        H --> I[Embedding + pHash]
        I --> J[Consulta FAISS/metadata]
        J --> K[Matches JSON]
    end

    K --> L[Redis: results]
    L --> M[Redis: status completed]
    M --> N[WebSocket eventos]
    M -.-> O[GET /v1/jobs]
```

- Leia em Obsidian para ver os diagramas; ajustável conforme necessidade.
