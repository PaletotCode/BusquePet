package phash

import (
	"bytes"
	"errors"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"
	"strings"

	"github.com/corona10/goimagehash"
)

// Engine computes perceptual hashes compatible with the Python pipeline.
type Engine struct {
	width  int
	height int
}

// NewEngine builds a pHash engine with the given hash size (defaults to 16x16).
func NewEngine(hashSize int) *Engine {
	if hashSize <= 0 {
		hashSize = 16
	}
	return &Engine{
		width:  hashSize,
		height: hashSize,
	}
}

// FromPath loads an image from disk and returns its perceptual hash.
func (e *Engine) FromPath(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	img, _, err := image.Decode(file)
	if err != nil {
		return "", err
	}
	return e.fromImage(img)
}

// FromBytes decodes the provided buffer and returns its perceptual hash.
func (e *Engine) FromBytes(data []byte) (string, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return "", err
	}
	return e.fromImage(img)
}

func (e *Engine) fromImage(img image.Image) (string, error) {
	if img == nil {
		return "", errors.New("phash: image is nil")
	}
	hash, err := goimagehash.ExtPerceptionHash(img, e.width, e.height)
	if err != nil {
		return "", err
	}
	hashStr := hash.ToString()
	parts := strings.SplitN(hashStr, ":", 2)
	if len(parts) == 2 {
		return parts[1], nil
	}
	return hashStr, nil
}
