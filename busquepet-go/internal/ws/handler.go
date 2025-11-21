package ws

import (
	"net/http"

	"github.com/gorilla/websocket"
	"go.uber.org/zap"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

// Handler upgrades HTTP connections to WebSocket and registers the client.
func Handler(hub *Hub) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			if hub != nil && hub.logger != nil {
				hub.logger.Warn("failed to upgrade websocket", zap.Error(err))
			}
			return
		}
		client := &client{
			hub:  hub,
			conn: conn,
			send: make(chan []byte, 256),
		}
		hub.register <- client
		go client.writePump()
		go client.readPump()
	}
}
