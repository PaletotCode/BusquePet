package config

import (
	"time"

	"github.com/kelseyhightower/envconfig"
)

// Config groups every environment-driven setting required by the Go services.
type Config struct {
	Port             string        `envconfig:"PORT" default:"8080"`
	RedisURL         string        `envconfig:"REDIS_URL" default:"redis://localhost:6379/0"`
	NATSURL          string        `envconfig:"NATS_URL" default:"nats://localhost:4222"`
	PythonWebhookURL string        `envconfig:"PYTHON_WEBHOOK_URL" required:"true"`
	StorageDir       string        `envconfig:"STORAGE_DIR" default:"./processed"`
	MaxImageMB       int64         `envconfig:"MAX_IMAGE_MB" default:"12"`
	RequestTimeout   time.Duration `envconfig:"REQUEST_TIMEOUT" default:"15s"`
}

// Load builds the Config from environment variables and validates required fields.
func Load() (Config, error) {
	var cfg Config
	err := envconfig.Process("BUSQUEPET", &cfg)
	return cfg, err
}
