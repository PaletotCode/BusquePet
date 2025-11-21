package errors

import (
	"fmt"
	"net/http"
)

// APIError standardizes error messages returned by HTTP handlers.
type APIError struct {
	Code    int    `json:"-"`
	Message string `json:"message"`
	Err     error  `json:"-"`
}

func (a APIError) Error() string {
	if a.Err != nil {
		return fmt.Sprintf("%s: %v", a.Message, a.Err)
	}
	return a.Message
}

// BadRequest represents a validation failure (HTTP 400).
func BadRequest(message string, err error) APIError {
	return APIError{Code: http.StatusBadRequest, Message: message, Err: err}
}

// Internal represents an unexpected exception (HTTP 500).
func Internal(message string, err error) APIError {
	return APIError{Code: http.StatusInternalServerError, Message: message, Err: err}
}
