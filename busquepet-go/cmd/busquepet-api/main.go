package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/cors"
	"go.uber.org/zap"

	"github.com/busquepet/busquepet-go/internal/cache"
	"github.com/busquepet/busquepet-go/internal/phash"
	"github.com/busquepet/busquepet-go/internal/preprocess"
	"github.com/busquepet/busquepet-go/internal/queue"
	"github.com/busquepet/busquepet-go/internal/ws"
	"github.com/busquepet/busquepet-go/pkg/config"
	apperrors "github.com/busquepet/busquepet-go/pkg/errors"
	"github.com/busquepet/busquepet-go/pkg/logger"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	cfg, err := config.Load()
	if err != nil {
		panic(fmt.Errorf("load config: %w", err))
	}

	log, err := logger.New()
	if err != nil {
		panic(fmt.Errorf("init logger: %w", err))
	}
	defer log.Sync() //nolint:errcheck

	preproc, err := preprocess.NewService(log, cfg.StorageDir, cfg.MaxImageMB)
	if err != nil {
		log.Fatal("init preprocess service", zap.Error(err))
	}
	phashEngine := phash.NewEngine(16)

	redisCache, err := cache.NewRedisCache(cfg.RedisURL)
	if err != nil {
		log.Fatal("connect redis", zap.Error(err))
	}
	defer redisCache.Close()

	natsQueue, err := queue.NewNATSQueue(cfg.NATSURL, log)
	if err != nil {
		log.Fatal("connect nats", zap.Error(err))
	}
	defer natsQueue.Close()

	hub := ws.NewHub(log)
	go hub.Run(ctx)

	worker := NewIngestWorker(cfg, natsQueue, redisCache, hub, log)
	go func() {
		if err := worker.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
			log.Error("worker stopped", zap.Error(err))
			stop()
		}
	}()

	server := NewServer(cfg, log, preproc, phashEngine, redisCache, natsQueue, hub)
	if err := server.Start(ctx); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal("http server exited with error", zap.Error(err))
	}
}

// Server wires dependencies together and exposes HTTP routes.
type Server struct {
	cfg          config.Config
	logger       *zap.Logger
	preprocessor *preprocess.Service
	phash        *phash.Engine
	cache        *cache.RedisCache
	queue        *queue.NATSQueue
	hub          *ws.Hub
}

// NewServer returns a configured Server instance.
func NewServer(
	cfg config.Config,
	logger *zap.Logger,
	preproc *preprocess.Service,
	phashEngine *phash.Engine,
	cache *cache.RedisCache,
	queue *queue.NATSQueue,
	hub *ws.Hub,
) *Server {
	return &Server{
		cfg:          cfg,
		logger:       logger,
		preprocessor: preproc,
		phash:        phashEngine,
		cache:        cache,
		queue:        queue,
		hub:          hub,
	}
}

// Start launches the HTTP server.
func (s *Server) Start(ctx context.Context) error {
	router := chi.NewRouter()
	router.Use(
		middleware.RequestID,
		middleware.RealIP,
		middleware.Recoverer,
		middleware.Timeout(s.cfg.RequestTimeout),
		s.logRequests(),
		cors.Handler(cors.Options{
			AllowedOrigins:   []string{"*"},
			AllowedMethods:   []string{http.MethodGet, http.MethodPost, http.MethodOptions},
			AllowedHeaders:   []string{"*"},
			AllowCredentials: false,
			MaxAge:           300,
		}),
	)
	// Servir arquivos do workspace (para exibir imagens dos matches no frontend)
	fileServer := http.StripPrefix("/files/", http.FileServer(http.Dir(".")))
	router.Handle("/files/*", fileServer)
	router.Get("/healthz", s.handleHealth)
	router.Route("/v1", func(r chi.Router) {
		r.Post("/ingest", s.handleIngest)
		r.Get("/jobs/{jobID}", s.handleGetJob)
		r.Get("/ws", ws.Handler(s.hub))
	})

	srv := &http.Server{
		Addr:         ":" + s.cfg.Port,
		Handler:      router,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 60 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := srv.Shutdown(shutdownCtx); err != nil {
			s.logger.Warn("http shutdown error", zap.Error(err))
		}
	}()

	s.logger.Info("BusquePet API listening", zap.String("addr", srv.Addr))
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	return nil
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

func (s *Server) handleIngest(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		s.respondError(w, apperrors.BadRequest("multipart parse failed", err))
		return
	}
	file, header, err := r.FormFile("image")
	if err != nil {
		s.respondError(w, apperrors.BadRequest("missing image field", err))
		return
	}
	defer file.Close()

	result, err := s.preprocessor.Process(r.Context(), file, header.Filename)
	if err != nil {
		switch {
		case errors.Is(err, preprocess.ErrImageTooLarge):
			s.respondError(w, apperrors.BadRequest("image too large", err))
		case errors.Is(err, preprocess.ErrInvalidImage):
			s.respondError(w, apperrors.BadRequest("invalid image", err))
		default:
			s.respondError(w, apperrors.Internal("preprocess failed", err))
		}
		return
	}

	hash, err := s.phash.FromPath(result.NormalizedPath)
	if err != nil {
		s.respondError(w, apperrors.Internal("phash failed", err))
		return
	}

	jobRecord := cache.JobRecord{
		JobID:            result.JobID,
		Status:           cache.StatusPending,
		OriginalFilename: result.OriginalFilename,
		ContentType:      result.ContentType,
		RawPath:          result.RawPath,
		NormalizedPath:   result.NormalizedPath,
		PHash:            hash,
		SizeBytes:        result.SizeBytes,
		Width:            result.Width,
		Height:           result.Height,
		Checksum:         result.Checksum,
		CreatedAt:        result.ProcessedAt,
		UpdatedAt:        result.ProcessedAt,
	}

	if err := s.cache.SaveJob(r.Context(), jobRecord); err != nil {
		s.respondError(w, apperrors.Internal("failed to persist job", err))
		return
	}

	msg := queue.IngestMessage{
		JobID:            result.JobID,
		NormalizedPath:   result.NormalizedPath,
		RawPath:          result.RawPath,
		PHash:            hash,
		SizeBytes:        result.SizeBytes,
		Width:            result.Width,
		Height:           result.Height,
		OriginalFilename: result.OriginalFilename,
		ContentType:      result.ContentType,
		Checksum:         result.Checksum,
	}
	if err := s.queue.PublishIngest(r.Context(), msg); err != nil {
		s.respondError(w, apperrors.Internal("queue publish failed", err))
		return
	}

	s.hub.Broadcast(statusMessage{
		Type:    "job.status",
		JobID:   result.JobID,
		Status:  string(cache.StatusPending),
		Message: "ingestion accepted",
	})

	resp := map[string]interface{}{
		"job_id":  result.JobID,
		"status":  cache.StatusPending,
		"phash":   hash,
		"message": "job enqueued",
	}
	writeJSON(w, http.StatusAccepted, resp)
}

func (s *Server) handleGetJob(w http.ResponseWriter, r *http.Request) {
	jobID := chi.URLParam(r, "jobID")
	jobID = strings.TrimSpace(jobID)
	if jobID == "" {
		s.respondError(w, apperrors.BadRequest("job_id is required", nil))
		return
	}

	record, err := s.cache.GetJob(r.Context(), jobID)
	if err != nil {
		if errors.Is(err, cache.ErrNotFound) {
			s.respondError(w, apperrors.APIError{Code: http.StatusNotFound, Message: "job not found", Err: err})
			return
		}
		s.respondError(w, apperrors.Internal("fetch job failed", err))
		return
	}
	response := map[string]interface{}{
		"job": record,
	}
	result, err := s.cache.GetResult(r.Context(), jobID)
	if err == nil {
		response["result"] = result
	} else if !errors.Is(err, cache.ErrNotFound) {
		s.logger.Warn("failed to fetch cached result", zap.Error(err))
	}
	writeJSON(w, http.StatusOK, response)
}

func (s *Server) logRequests() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			start := time.Now()
			ww := middleware.NewWrapResponseWriter(w, r.ProtoMajor)
			next.ServeHTTP(ww, r)
			s.logger.Info("http request",
				zap.String("method", r.Method),
				zap.String("path", r.URL.Path),
				zap.Int("status", ww.Status()),
				zap.Int("bytes", ww.BytesWritten()),
				zap.Duration("latency", time.Since(start)),
			)
		})
	}
}

func (s *Server) respondError(w http.ResponseWriter, apiErr apperrors.APIError) {
	if apiErr.Code == 0 {
		apiErr.Code = http.StatusInternalServerError
	}
	writeJSON(w, apiErr.Code, map[string]string{"error": apiErr.Message})
	s.logger.Warn("api error", zap.String("message", apiErr.Message), zap.Error(apiErr.Err))
}

func writeJSON(w http.ResponseWriter, status int, payload interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if payload == nil {
		return
	}
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(payload)
}

type statusMessage struct {
	Type    string `json:"type"`
	JobID   string `json:"job_id"`
	Status  string `json:"status"`
	Message string `json:"message,omitempty"`
}

// IngestWorker drains NATS subjects and coordinates Python calls.
type IngestWorker struct {
	cfg       config.Config
	queue     *queue.NATSQueue
	cache     *cache.RedisCache
	hub       *ws.Hub
	logger    *zap.Logger
	client    *http.Client
	pythonURL string
}

// NewIngestWorker configures a worker.
func NewIngestWorker(
	cfg config.Config,
	queue *queue.NATSQueue,
	cache *cache.RedisCache,
	hub *ws.Hub,
	logger *zap.Logger,
) *IngestWorker {
	client := &http.Client{
		Timeout: cfg.RequestTimeout,
	}
	return &IngestWorker{
		cfg:       cfg,
		queue:     queue,
		cache:     cache,
		hub:       hub,
		logger:    logger,
		client:    client,
		pythonURL: cfg.PythonWebhookURL,
	}
}

// Run blocks until context cancellation.
func (w *IngestWorker) Run(ctx context.Context) error {
	if w.pythonURL == "" {
		return fmt.Errorf("python webhook URL is empty")
	}
	return w.queue.ConsumeIngest(ctx, w.process)
}

func (w *IngestWorker) process(ctx context.Context, msg queue.IngestMessage) error {
	record, err := w.cache.UpdateJobStatus(ctx, msg.JobID, cache.StatusProcessing, "")
	if err != nil {
		return err
	}
	w.notify(record, "job started")

	response, err := w.invokePython(ctx, msg)
	if err != nil {
		w.cache.UpdateJobStatus(ctx, msg.JobID, cache.StatusFailed, err.Error()) //nolint:errcheck
		w.notify(cache.JobRecord{
			JobID:  msg.JobID,
			Status: cache.StatusFailed,
			Error:  err.Error(),
		}, "python call failed")
		return err
	}

	result := cache.SearchResult{
		JobID:       msg.JobID,
		Matches:     make([]cache.Match, len(response.Matches)),
		RetrievedAt: time.Now().UTC(),
		Metadata:    response.Metadata,
	}
	for i, match := range response.Matches {
		result.Matches[i] = cache.Match{
			ImageID:   match.ImageID,
			Breed:     match.Breed,
			Score:     match.Score,
			Distance:  match.Distance,
			ImagePath: match.ImagePath,
			PHash:     match.PHash,
		}
	}
	if err := w.cache.SaveResult(ctx, result); err != nil {
		return err
	}
	record, err = w.cache.UpdateJobStatus(ctx, msg.JobID, cache.StatusCompleted, "")
	if err != nil {
		return err
	}
	w.notify(record, "job completed")
	w.hub.Broadcast(map[string]interface{}{
		"type":   "job.result",
		"job_id": result.JobID,
		"result": result,
	})
	return nil
}

func (w *IngestWorker) invokePython(ctx context.Context, msg queue.IngestMessage) (queue.PythonResponse, error) {
	payload := struct {
		JobID    string                 `json:"job_id"`
		Image    string                 `json:"image_path"`
		PHash    string                 `json:"phash"`
		Metadata map[string]interface{} `json:"metadata"`
	}{
		JobID: msg.JobID,
		Image: filepath.Clean(msg.NormalizedPath),
		PHash: msg.PHash,
		Metadata: map[string]interface{}{
			"raw_path":           filepath.Clean(msg.RawPath),
			"original_filename":  msg.OriginalFilename,
			"content_type":       msg.ContentType,
			"width":              msg.Width,
			"height":             msg.Height,
			"size_bytes":         msg.SizeBytes,
			"checksum":           msg.Checksum,
			"received_at":        time.Now().UTC().Format(time.RFC3339),
			"phash_source":       "go",
			"normalized_storage": filepath.Dir(filepath.Clean(msg.NormalizedPath)),
		},
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return queue.PythonResponse{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, w.pythonURL, bytes.NewReader(body))
	if err != nil {
		return queue.PythonResponse{}, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := w.client.Do(req)
	if err != nil {
		return queue.PythonResponse{}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= http.StatusBadRequest {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return queue.PythonResponse{}, fmt.Errorf("python returned %d: %s", resp.StatusCode, string(b))
	}

	var parsed queue.PythonResponse
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		return queue.PythonResponse{}, err
	}
	if parsed.JobID == "" {
		parsed.JobID = msg.JobID
	}
	return parsed, nil
}

func (w *IngestWorker) notify(record cache.JobRecord, message string) {
	w.hub.Broadcast(statusMessage{
		Type:    "job.status",
		JobID:   record.JobID,
		Status:  string(record.Status),
		Message: message,
	})
}
