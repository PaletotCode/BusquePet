package preprocess

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/disintegration/imaging"
	"github.com/google/uuid"
	"go.uber.org/zap"
)

const (
	rawDirName        = "raw"
	normalizedDirName = "normalized"
	defaultImageSize  = 224
)

var (
	// ErrImageTooLarge indicates the payload exceeded BUSQUEPET_MAX_IMAGE_MB.
	ErrImageTooLarge = errors.New("preprocess: image exceeds maximum allowed size")
	// ErrInvalidImage indicates an unreadable or unsupported image.
	ErrInvalidImage = errors.New("preprocess: invalid image")
)

// Service validates, normalizes and persists uploads.
type Service struct {
	logger    *zap.Logger
	storage   string
	maxBytes  int64
	imageSize int
}

// Result captures metadata about a processed upload.
type Result struct {
	JobID            string    `json:"job_id"`
	OriginalFilename string    `json:"original_filename"`
	ContentType      string    `json:"content_type"`
	RawPath          string    `json:"raw_path"`
	NormalizedPath   string    `json:"normalized_path"`
	SizeBytes        int64     `json:"size_bytes"`
	Width            int       `json:"width"`
	Height           int       `json:"height"`
	Checksum         string    `json:"checksum"`
	ProcessedAt      time.Time `json:"processed_at"`
}

// NewService builds a Service that will store assets under storageDir.
func NewService(logger *zap.Logger, storageDir string, maxImageMB int64) (*Service, error) {
	if logger == nil {
		return nil, errors.New("preprocess: logger is required")
	}
	if storageDir == "" {
		return nil, errors.New("preprocess: storage directory is required")
	}
	absDir, err := filepath.Abs(storageDir)
	if err != nil {
		return nil, fmt.Errorf("preprocess: resolve storage dir: %w", err)
	}
	if err := os.MkdirAll(filepath.Join(absDir, rawDirName), 0o755); err != nil {
		return nil, fmt.Errorf("preprocess: create raw dir: %w", err)
	}
	if err := os.MkdirAll(filepath.Join(absDir, normalizedDirName), 0o755); err != nil {
		return nil, fmt.Errorf("preprocess: create normalized dir: %w", err)
	}
	maxBytes := maxImageMB * 1024 * 1024
	if maxBytes <= 0 {
		maxBytes = 5 * 1024 * 1024
	}
	return &Service{
		logger:    logger,
		storage:   absDir,
		maxBytes:  maxBytes,
		imageSize: defaultImageSize,
	}, nil
}

// Process consumes the reader, validates and persists the upload.
func (s *Service) Process(ctx context.Context, reader io.Reader, filename string) (Result, error) {
	if err := ctx.Err(); err != nil {
		return Result{}, err
	}

	jobID := uuid.NewString()
	data, size, err := s.readWithLimit(reader)
	if err != nil {
		return Result{}, err
	}

	contentType := http.DetectContentType(data[:min(len(data), 512)])
	img, format, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return Result{}, fmt.Errorf("%w: %v", ErrInvalidImage, err)
	}

	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	normalized := imaging.Resize(img, s.imageSize, s.imageSize, imaging.Lanczos)
	rawPath, normalizedPath, err := s.persist(jobID, filename, format, data, normalized)
	if err != nil {
		return Result{}, err
	}

	sum := sha256.Sum256(data)
	checksum := hex.EncodeToString(sum[:])
	result := Result{
		JobID:            jobID,
		OriginalFilename: sanitizeFilename(filename),
		ContentType:      contentType,
		RawPath:          rawPath,
		NormalizedPath:   normalizedPath,
		SizeBytes:        size,
		Width:            width,
		Height:           height,
		Checksum:         checksum,
		ProcessedAt:      time.Now().UTC(),
	}
	return result, nil
}

func (s *Service) readWithLimit(reader io.Reader) ([]byte, int64, error) {
	var buf bytes.Buffer
	limited := &io.LimitedReader{R: reader, N: s.maxBytes + 1}
	written, err := io.Copy(&buf, limited)
	if err != nil {
		return nil, 0, fmt.Errorf("preprocess: read payload: %w", err)
	}
	if limited.N <= 0 {
		return nil, 0, ErrImageTooLarge
	}
	return buf.Bytes(), written, nil
}

func (s *Service) persist(jobID, filename, format string, rawData []byte, normalized image.Image) (string, string, error) {
	rawExt := strings.ToLower(filepath.Ext(filename))
	if rawExt == "" && format != "" {
		rawExt = "." + strings.ToLower(format)
	}
	if rawExt == "" {
		rawExt = ".bin"
	}
	rawPath := filepath.Join(s.storage, rawDirName, fmt.Sprintf("%s%s", jobID, rawExt))
	if err := os.WriteFile(rawPath, rawData, 0o644); err != nil {
		return "", "", fmt.Errorf("preprocess: persist raw: %w", err)
	}

	normPath := filepath.Join(s.storage, normalizedDirName, fmt.Sprintf("%s.jpg", jobID))
	file, err := os.Create(normPath)
	if err != nil {
		return "", "", fmt.Errorf("preprocess: create normalized file: %w", err)
	}
	defer file.Close()

	if err := imaging.Encode(file, normalized, imaging.JPEG, imaging.JPEGQuality(95)); err != nil {
		return "", "", fmt.Errorf("preprocess: encode normalized: %w", err)
	}
	return rawPath, normPath, nil
}

func sanitizeFilename(name string) string {
	name = filepath.Base(strings.TrimSpace(name))
	name = strings.ReplaceAll(name, " ", "_")
	if name == "." || name == "/" {
		return "unknown"
	}
	return name
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
