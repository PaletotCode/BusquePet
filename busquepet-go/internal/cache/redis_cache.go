package cache

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"
)

const (
	jobTTL     = 48 * time.Hour
	resultTTL  = 48 * time.Hour
	namespace  = "busquepet"
	jobsKey    = namespace + ":jobs:"
	resultsKey = namespace + ":results:"
)

// ErrNotFound signals the requested key is absent.
var ErrNotFound = errors.New("cache: not found")

// JobStatus describes the state of a processing request.
type JobStatus string

const (
	StatusPending    JobStatus = "pending"
	StatusProcessing JobStatus = "processing"
	StatusCompleted  JobStatus = "completed"
	StatusFailed     JobStatus = "failed"
)

// JobRecord stores metadata for an image ingestion request.
type JobRecord struct {
	JobID            string    `json:"job_id"`
	Status           JobStatus `json:"status"`
	OriginalFilename string    `json:"original_filename"`
	ContentType      string    `json:"content_type"`
	RawPath          string    `json:"raw_path"`
	NormalizedPath   string    `json:"normalized_path"`
	PHash            string    `json:"phash"`
	SizeBytes        int64     `json:"size_bytes"`
	Width            int       `json:"width"`
	Height           int       `json:"height"`
	Checksum         string    `json:"checksum"`
	Error            string    `json:"error,omitempty"`
	CreatedAt        time.Time `json:"created_at"`
	UpdatedAt        time.Time `json:"updated_at"`
}

// Match represents a single hybrid search result coming from Python.
type Match struct {
	ImageID   string  `json:"image_id"`
	Breed     string  `json:"breed"`
	Score     float64 `json:"score"`
	Distance  float64 `json:"distance"`
	ImagePath string  `json:"image_path"`
	PHash     string  `json:"phash,omitempty"`
}

// SearchResult stores final matches returned by Python.
type SearchResult struct {
	JobID       string            `json:"job_id"`
	Matches     []Match           `json:"matches"`
	RetrievedAt time.Time         `json:"retrieved_at"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// RedisCache provides typed helpers on top of redis.Client.
type RedisCache struct {
	client *redis.Client
}

// NewRedisCache connects to Redis and validates connectivity.
func NewRedisCache(redisURL string) (*RedisCache, error) {
	if redisURL == "" {
		return nil, errors.New("cache: redis URL is empty")
	}
	opt, err := redis.ParseURL(redisURL)
	if err != nil {
		return nil, fmt.Errorf("cache: parse url: %w", err)
	}
	client := redis.NewClient(opt)
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := client.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("cache: ping redis: %w", err)
	}
	return &RedisCache{client: client}, nil
}

// Close closes the underlying Redis connection.
func (c *RedisCache) Close() error {
	return c.client.Close()
}

// SaveJob stores or replaces the job record.
func (c *RedisCache) SaveJob(ctx context.Context, record JobRecord) error {
	record.UpdatedAt = time.Now().UTC()
	if record.CreatedAt.IsZero() {
		record.CreatedAt = record.UpdatedAt
	}
	payload, err := json.Marshal(record)
	if err != nil {
		return err
	}
	return c.client.Set(ctx, jobsKey+record.JobID, payload, jobTTL).Err()
}

// UpdateJobStatus updates the status and optional error message.
func (c *RedisCache) UpdateJobStatus(ctx context.Context, jobID string, status JobStatus, errMsg string) (JobRecord, error) {
	record, err := c.GetJob(ctx, jobID)
	if err != nil {
		return JobRecord{}, err
	}
	record.Status = status
	record.Error = errMsg
	record.UpdatedAt = time.Now().UTC()
	if err := c.SaveJob(ctx, record); err != nil {
		return JobRecord{}, err
	}
	return record, nil
}

// GetJob fetches a job by ID.
func (c *RedisCache) GetJob(ctx context.Context, jobID string) (JobRecord, error) {
	val, err := c.client.Get(ctx, jobsKey+jobID).Result()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return JobRecord{}, ErrNotFound
		}
		return JobRecord{}, err
	}
	var record JobRecord
	if err := json.Unmarshal([]byte(val), &record); err != nil {
		return JobRecord{}, err
	}
	return record, nil
}

// SaveResult stores the matches associated with a job.
func (c *RedisCache) SaveResult(ctx context.Context, result SearchResult) error {
	payload, err := json.Marshal(result)
	if err != nil {
		return err
	}
	return c.client.Set(ctx, resultsKey+result.JobID, payload, resultTTL).Err()
}

// GetResult fetches stored matches for a job.
func (c *RedisCache) GetResult(ctx context.Context, jobID string) (SearchResult, error) {
	val, err := c.client.Get(ctx, resultsKey+jobID).Result()
	if err != nil {
		if errors.Is(err, redis.Nil) {
			return SearchResult{}, ErrNotFound
		}
		return SearchResult{}, err
	}
	var result SearchResult
	if err := json.Unmarshal([]byte(val), &result); err != nil {
		return SearchResult{}, err
	}
	return result, nil
}
