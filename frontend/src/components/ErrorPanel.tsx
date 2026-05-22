import { Trash2, X } from "lucide-react";
import type { Track } from "../types";

type Props = {
  tracks: Track[];
  onDismiss: (track: Track) => void;
  onDelete: (track: Track) => void;
};

export function ErrorPanel({ tracks, onDismiss, onDelete }: Props) {
  if (tracks.length === 0) {
    return null;
  }

  return (
    <section className="error-panel" aria-label="track errors">
      {tracks.map((track) => (
        <div className="error-line" key={track.id}>
          <div>
            <strong>{track.title}</strong>
            <span>{track.error ?? "Unknown error"}</span>
          </div>
          <div className="error-actions">
            <button
              type="button"
              className="icon-button"
              aria-label={`Dismiss ${track.title}`}
              title="Dismiss"
              onClick={() => onDismiss(track)}
            >
              <X aria-hidden="true" size={16} strokeWidth={2.4} />
            </button>
            <button
              type="button"
              className="icon-button danger-button"
              aria-label={`Delete ${track.title}`}
              title="Delete"
              onClick={() => onDelete(track)}
            >
              <Trash2 aria-hidden="true" size={16} strokeWidth={2.4} />
            </button>
          </div>
        </div>
      ))}
    </section>
  );
}
