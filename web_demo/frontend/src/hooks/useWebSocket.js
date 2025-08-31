import { useState, useCallback, useRef } from 'react';

/**
 * 简化的WebSocket连接钩子 (v2 - 手动控制)
 *
 * @param {string} url - WebSocket连接URL
 * @param {Object} callbacks - 回调函数
 */
const useWebSocket = (url, { onOpen, onMessage, onClose, onError }) => {
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState(null);
  const socketRef = useRef(null);

  const connect = useCallback(() => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      console.log('WebSocket已经连接');
      return;
    }

    console.log('尝试连接WebSocket...');
    const socket = new WebSocket(url);
    socketRef.current = socket;

    socket.onopen = (event) => {
      console.log('WebSocket连接成功打开');
      setIsConnected(true);
      setError(null);
      onOpen?.(event);
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage?.(data);
      } catch (err) {
        console.error('解析消息失败', err);
      }
    };

    socket.onclose = (event) => {
      console.log(`WebSocket连接关闭, Code: ${event.code}`);
      setIsConnected(false);
      socketRef.current = null;
      onClose?.(event);
    };

    socket.onerror = (event) => {
      console.error('WebSocket发生错误', event);
      setError('WebSocket连接发生错误');
      onError?.(event);
    };
  }, [url, onOpen, onMessage, onClose, onError]);

  const disconnect = useCallback(() => {
    if (socketRef.current) {
      console.log('手动断开WebSocket连接...');
      socketRef.current.close(1000, "用户主动断开");
      socketRef.current = null;
      setIsConnected(false);
    }
  }, []);

  const sendMessage = useCallback((message) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(JSON.stringify(message));
      return true;
    }
    return false;
  }, []);

  const sendBinary = useCallback((data) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(data);
      return true;
    }
    return false;
  }, []);

  return {
    isConnected,
    error,
    connect,
    disconnect,
    sendMessage,
    sendBinary,
  };
};

export default useWebSocket;