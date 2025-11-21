package queue

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/nats-io/nats.go"
	"go.uber.org/zap"
)

const (
	ingestSubject  = "busquepet.ingest"
	consumerGroup  = "busquepet-workers"
	handlerTimeout = 60 * time.Second
)

// IngestMessage represents the work that must be executed by the async worker.
type IngestMessage struct {
	JobID            string `json:"job_id"`
	NormalizedPath   string `json:"normalized_path"`
	RawPath          string `json:"raw_path"`
	PHash            string `json:"phash"`
	SizeBytes        int64  `json:"size_bytes"`
	Width            int    `json:"width"`
	Height           int    `json:"height"`
	OriginalFilename string `json:"original_filename"`
	ContentType      string `json:"content_type"`
	Checksum         string `json:"checksum"`
}

// PythonMatch describes the response for each retrieved image provided by Python.
type PythonMatch struct {
	ImageID   string  `json:"image_id"`
	Breed     string  `json:"breed"`
	Score     float64 `json:"score"`
	Distance  float64 `json:"distance"`
	ImagePath string  `json:"image_path"`
	PHash     string  `json:"phash,omitempty"`
}

// PythonResponse is the body expected from the Python webhook.
type PythonResponse struct {
	JobID       string            `json:"job_id"`
	Matches     []PythonMatch     `json:"matches"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	DurationMS  int64             `json:"duration_ms,omitempty"`
	Description string            `json:"description,omitempty"`
}

// NATSQueue encapsulates the connection and subjects used by BusquePet.
type NATSQueue struct {
	conn    *nats.Conn
	logger  *zap.Logger
	subject string
}

// NewNATSQueue dials a NATS cluster and prepares the ingest subject.
func NewNATSQueue(url string, logger *zap.Logger) (*NATSQueue, error) {
	if url == "" {
		return nil, errors.New("queue: NATS URL is empty")
	}
	conn, err := nats.Connect(url,
		nats.Name("busquepet-api"),
		nats.ReconnectBufSize(2*1024*1024),
		nats.MaxReconnects(-1),
	)
	if err != nil {
		return nil, fmt.Errorf("queue: connect: %w", err)
	}
	return &NATSQueue{conn: conn, logger: logger, subject: ingestSubject}, nil
}

// Close shuts down the NATS connection.
func (q *NATSQueue) Close() {
	if q.conn != nil && !q.conn.IsClosed() {
		q.conn.Drain()
		q.conn.Close()
	}
}

// PublishIngest sends a message to the queue with standard JSON encoding.
func (q *NATSQueue) PublishIngest(ctx context.Context, msg IngestMessage) error {
	payload, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	return q.conn.PublishMsg(&nats.Msg{
		Subject: q.subject,
		Data:    payload,
	})
}

// ConsumeIngest registers a handler that receives ingest events.
func (q *NATSQueue) ConsumeIngest(ctx context.Context, handler func(context.Context, IngestMessage) error) error {
	if handler == nil {
		return errors.New("queue: handler is nil")
	}
	sub, err := q.conn.QueueSubscribe(q.subject, consumerGroup, func(msg *nats.Msg) {
		var payload IngestMessage
		if err := json.Unmarshal(msg.Data, &payload); err != nil {
			q.logger.Warn("discarding malformed ingest message", zap.Error(err))
			return
		}
		hCtx, cancel := context.WithTimeout(ctx, handlerTimeout)
		defer cancel()
		if err := handler(hCtx, payload); err != nil {
			q.logger.Error("ingest handler failed", zap.String("job_id", payload.JobID), zap.Error(err))
		}
	})
	if err != nil {
		return err
	}
	if err := q.conn.Flush(); err != nil {
		return err
	}
	q.logger.Info("nats queue consumer registered", zap.String("subject", q.subject))
	<-ctx.Done()
	done := make(chan struct{})
	go func() {
		if err := sub.Drain(); err != nil {
			q.logger.Warn("failed to drain nats subscription", zap.Error(err))
		}
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		q.logger.Warn("timeout draining nats subscription")
	}
	return nil
}
