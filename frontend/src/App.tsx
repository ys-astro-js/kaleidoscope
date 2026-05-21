import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { audioUrl, deleteAllTracks, deleteTrack, fetchTracks, uploadTracks } from "./api";
import MusicScene from "./MusicScene";
import type { Track } from "./types";

export default function App() {
  const [tracks, setTracks] = useState<Track[]>([]);
  const [selected, setSelected] = useState<Track | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [message, setMessage] = useState("");
  const [dismissedErrorIds, setDismissedErrorIds] = useState<Set<string>>(() => new Set());
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const loadTracks = useCallback(async () => {
    const nextTracks = await fetchTracks();
    setTracks(nextTracks);
    setSelected((current) => {
      if (!current) {
        return null;
      }
      return nextTracks.find((track) => track.id === current.id) ?? null;
    });
  }, []);

  useEffect(() => {
    loadTracks().catch(() => setMessage("연결 실패"));
    const interval = window.setInterval(() => {
      loadTracks().catch(() => setMessage("연결 실패"));
    }, 2500);
    return () => window.clearInterval(interval);
  }, [loadTracks]);

  const activeTracks = useMemo(
    () => tracks.filter((track) => track.status === "queued" || track.status === "processing"),
    [tracks]
  );
  const failedTracks = useMemo(() => tracks.filter((track) => track.status === "error"), [tracks]);
  const visibleFailedTracks = useMemo(
    () => failedTracks.filter((track) => !dismissedErrorIds.has(track.id)),
    [dismissedErrorIds, failedTracks]
  );

  const handleFiles = async (files: FileList | File[]) => {
    const selectedFiles = Array.from(files).filter(isAudioFile);
    if (selectedFiles.length === 0) {
      setMessage("오디오 파일 없음");
      return;
    }
    setMessage("업로드 중");
    await uploadTracks(selectedFiles);
    setMessage("");
    await loadTracks();
  };

  const playTrack = (track: Track) => {
    setSelected(track);
    const audio = audioRef.current;
    if (!audio) {
      return;
    }
    audio.src = audioUrl(track.id);
    audio.onloadedmetadata = () => {
      audio.currentTime = Math.min(39, Math.max(0, audio.duration - 1));
      void audio.play();
    };
  };

  const removeTrack = async (track: Track) => {
    await deleteTrack(track.id);
    if (selected?.id === track.id) {
      audioRef.current?.pause();
      setSelected(null);
    }
    setDismissedErrorIds((current) => {
      const next = new Set(current);
      next.delete(track.id);
      return next;
    });
    await loadTracks();
  };

  const removeAllTracks = async () => {
    await deleteAllTracks();
    audioRef.current?.pause();
    setSelected(null);
    setDismissedErrorIds(new Set());
    await loadTracks();
  };

  return (
    <main
      className={isDragging ? "app dragging" : "app"}
      onDragOver={(event) => {
        event.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setIsDragging(false);
        void handleFiles(event.dataTransfer.files);
      }}
    >
      <section className="topbar">
        <button type="button" onClick={() => fileInputRef.current?.click()}>
          Upload
        </button>
        {tracks.length > 0 ? (
          <button type="button" onClick={() => void removeAllTracks()}>
            Clear
          </button>
        ) : null}
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*"
          multiple
          onChange={(event) => {
            if (event.target.files) {
              void handleFiles(event.target.files);
              event.target.value = "";
            }
          }}
        />
        {activeTracks.map((track) => (
          <span className="skeleton skeleton-chip" key={track.id} aria-label={`${track.title} processing`} />
        ))}
        {message ? <span className="status-pill">{message}</span> : null}
      </section>

      <MusicScene tracks={tracks} selectedId={selected?.id ?? null} onSelect={playTrack} />

      {activeTracks.length > 0 ? (
        <section className="skeleton-stage" aria-label="processing tracks">
          {activeTracks.map((track) => (
            <div className="skeleton skeleton-art" key={track.id} />
          ))}
        </section>
      ) : null}

      {visibleFailedTracks.length > 0 ? (
        <section className="error-panel" aria-label="track errors">
          {visibleFailedTracks.map((track) => (
            <div className="error-line" key={track.id}>
              <div>
                <strong>{track.title}</strong>
                <span>{track.error ?? "Unknown error"}</span>
              </div>
              <div className="error-actions">
                <button
                  type="button"
                  onClick={() =>
                    setDismissedErrorIds((current) => {
                      const next = new Set(current);
                      next.add(track.id);
                      return next;
                    })
                  }
                >
                  Dismiss
                </button>
                <button type="button" onClick={() => void removeTrack(track)}>
                  Delete
                </button>
              </div>
            </div>
          ))}
        </section>
      ) : null}

      <section className="now-playing">
        <div>
          <p>{selected?.title ?? ""}</p>
          <small>{selected?.artist ?? selected?.filename ?? ""}</small>
        </div>
        {selected ? (
          <button type="button" onClick={() => void removeTrack(selected)}>
            Delete
          </button>
        ) : null}
        <audio ref={audioRef} controls />
      </section>
    </main>
  );
}

function isAudioFile(file: File): boolean {
  if (file.type.startsWith("audio/")) {
    return true;
  }
  return /\.(aac|flac|m4a|mp3|ogg|wav)$/i.test(file.name);
}
