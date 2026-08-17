class LiveWebSocketManager {
  constructor() {
    this.ws = null;
    this.url = null;
    this.dataListeners = new Set();
    this.statusListeners = new Set();
    
    this.retryDelay = 1000;
    this.maxRetryDelay = 30000;
    this.reconnectTimer = null;
    this.isExplicitDisconnect = false;
    this.connectionState = 'disconnected'; // 'connected' | 'reconnecting' | 'disconnected'
  }

  getWsUrl() {
    if (import.meta.env.VITE_WS_URL) {
      return import.meta.env.VITE_WS_URL;
    }
    const apiBase = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    
    try {
      const parsed = new URL(apiBase);
      const host = parsed.host;
      return `${wsProtocol}//${host}/ws/live`;
    } catch {
      return 'ws://localhost:8000/ws/live';
    }
  }

  connect() {
    this.isExplicitDisconnect = false;
    this.url = this.getWsUrl();

    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this._setConnectionState(this.retryDelay > 1000 ? 'reconnecting' : 'disconnected');

    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        this.retryDelay = 1000;
        this._setConnectionState('connected');
      };

      this.ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          this._notifyDataListeners(payload);
        } catch (err) {
          console.error('[WebSocket] Failed to parse message:', err);
        }
      };

      this.ws.onerror = (error) => {
        console.warn('[WebSocket] Connection error:', error);
      };

      this.ws.onclose = () => {
        if (!this.isExplicitDisconnect) {
          this._setConnectionState('reconnecting');
          this._scheduleReconnect();
        } else {
          this._setConnectionState('disconnected');
        }
      };
    } catch (err) {
      console.error('[WebSocket] Failed to establish socket:', err);
      this._setConnectionState('reconnecting');
      this._scheduleReconnect();
    }
  }

  _scheduleReconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
    }

    this.reconnectTimer = setTimeout(() => {
      this.connect();
      // Exponential backoff
      this.retryDelay = Math.min(this.retryDelay * 2, this.maxRetryDelay);
    }, this.retryDelay);
  }

  disconnect() {
    this.isExplicitDisconnect = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this._setConnectionState('disconnected');
  }

  subscribeData(callback) {
    this.dataListeners.add(callback);
    return () => this.dataListeners.delete(callback);
  }

  subscribeStatus(callback) {
    this.statusListeners.add(callback);
    // Immediately inform subscriber of current state
    callback(this.connectionState);
    return () => this.statusListeners.delete(callback);
  }

  _setConnectionState(state) {
    this.connectionState = state;
    this.statusListeners.forEach((listener) => {
      try {
        listener(state);
      } catch (err) {
        console.error('[WebSocket] Status listener error:', err);
      }
    });
  }

  _notifyDataListeners(data) {
    this.dataListeners.forEach((listener) => {
      try {
        listener(data);
      } catch (err) {
        console.error('[WebSocket] Data listener error:', err);
      }
    });
  }
}

export const wsManager = new LiveWebSocketManager();
