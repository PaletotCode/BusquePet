FROM golang:1.22 AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o /busquepet ./cmd/busquepet-api

FROM gcr.io/distroless/base-debian12
WORKDIR /workspace
COPY --from=builder /busquepet /usr/local/bin/busquepet
ENV BUSQUEPET_PORT=8080
EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/busquepet"]
