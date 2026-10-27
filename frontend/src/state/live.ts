import { useCallback, useEffect, useRef, useState } from "react";
import type { Snapshot } from "../types/api";

export type ConnectionState = "connecting" | "live" | "offline";

function wsUrl(): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}/api/events`;
}

async function fetchSnapshot(): Promise<Snapshot> {
  const response = await fetch("/api/state");
  if (!response.ok) {
    throw new Error(`state request failed: ${response.status}`);
  }
  return (await response.json()) as Snapshot;
}

export function useLiveSnapshot(): {
  snapshot: Snapshot | null;
  connection: ConnectionState;
  refresh: () => void;
} {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const versionRef = useRef(-1);

  const accept = useCallback((next: Snapshot) => {
    if (next.state_version >= versionRef.current) {
      versionRef.current = next.state_version;
      setSnapshot(next);
    }
  }, []);

  const refresh = useCallback(() => {
    void fetchSnapshot().then(accept).catch(() => setConnection("offline"));
  }, [accept]);

  useEffect(() => {
    refresh();
    let socket: WebSocket | null = null;
    let retry: number | undefined;
    let closed = false;

    const connect = () => {
      setConnection("connecting");
      socket = new WebSocket(wsUrl());
      socket.onopen = () => setConnection("live");
      socket.onmessage = (event) => {
        const message = JSON.parse(event.data) as {
          event_type: string;
          payload: Record<string, unknown>;
        };
        if (message.event_type === "STATE_SNAPSHOT") {
          accept(message.payload as unknown as Snapshot);
        } else if (message.event_type === "STATE_DELTA") {
          accept(message.payload as unknown as Snapshot);
        }
      };
      socket.onclose = () => {
        setConnection("offline");
        if (!closed) {
          retry = window.setTimeout(connect, 1500);
        }
      };
      socket.onerror = () => socket?.close();
    };

    connect();
    return () => {
      closed = true;
      if (retry) window.clearTimeout(retry);
      socket?.close();
    };
  }, [accept, refresh]);

  return { snapshot, connection, refresh };
}
