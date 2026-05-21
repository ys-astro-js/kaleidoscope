import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  audioUrl,
  deleteAllTracks,
  deleteTrack,
  fetchTracks,
  submitFeedback,
  uploadTracks
} from "./api";
import MusicScene from "./MusicScene";
import type { FeedbackLabel, Track, ViewMode } from "./types";

export default function App() {
  const [tracks, setTracks] = useState<Track[]>([]);
  const [selected, setSelected] = useState<Track | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [message, setMessage] = useState("");
  const [viewMode, setViewMode] = useState<ViewMode>("3d");
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
    loadTracks().catch(() => setMessage("Connection failed"));
    const interval = window.setInterval(() => {
      loadTracks().catch(() => setMessage("Connection failed"));
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
  const similarTracks = useMemo(() => {
    if (!selected) {
      return [];
    }
    return selected.similar
      .map((similar) => ({
        score: similar.score,
        track: tracks.find((track) => track.id === similar.id)
      }))
      .filter((similar): similar is { score: number; track: Track } => Boolean(similar.track));
  }, [selected, tracks]);

  const handleFiles = async (files: FileList | File[]) => {
    const selectedFiles = Array.from(files).filter(isAudioFile);
    if (selectedFiles.length === 0) {
      setMessage("No audio files found");
      return;
    }
    setMessage("Uploading");
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

  const sendFeedback = async (candidate: Track, label: FeedbackLabel) => {
    if (!selected) {
      return;
    }
    await submitFeedback(selected.id, candidate.id, label);
    setMessage("Feedback saved");
    await loadTracks();
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
    const confirmed = window.confirm("Delete all tracks?");
    if (!confirmed) {
      return;
    }
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
        <div className="mode-toggle" role="group" aria-label="view mode">
          <button
            type="button"
            className={viewMode === "2d" ? "active" : ""}
            onClick={() => setViewMode("2d")}
          >
            2D
          </button>
          <button
            type="button"
            className={viewMode === "3d" ? "active" : ""}
            onClick={() => setViewMode("3d")}
          >
            3D
          </button>
        </div>
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

      <MusicScene
        tracks={tracks}
        selectedId={selected?.id ?? null}
        viewMode={viewMode}
        onSelect={playTrack}
      />

      {selected && similarTracks.length > 0 ? (
        <section className="feedback-panel" aria-label="similarity feedback">
          <h2>Similar tracks</h2>
          {similarTracks.map(({ score, track }) => (
            <div className="feedback-line" key={track.id}>
              <div>
                <strong>{track.title}</strong>
                <span>{Math.round(score * 100)}%</span>
              </div>
              <div className="feedback-actions">
                <button type="button" onClick={() => void sendFeedback(track, "similar")}>
                  Similar
                </button>
                <button type="button" onClick={() => void sendFeedback(track, "not_similar")}>
                  Not similar
                </button>
              </div>
            </div>
          ))}
        </section>
      ) : null}

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
