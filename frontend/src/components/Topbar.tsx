import { useRef } from "react";
import { CloudUpload, Trash2 } from "lucide-react";
import type { Track, ViewMode } from "../types";

type Props = {
  tracksCount: number;
  activeTracks: Track[];
  message: string;
  viewMode: ViewMode;
  onViewModeChange: (mode: ViewMode) => void;
  onClearTracks: () => void;
  onFilesSelected: (files: FileList) => void;
};

export function Topbar({
  tracksCount,
  activeTracks,
  message,
  viewMode,
  onViewModeChange,
  onClearTracks,
  onFilesSelected,
}: Props) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const nextViewMode = viewMode === "2d" ? "3d" : "2d";
  const viewModeLabel = viewMode === "2d" ? "2D" : "3D";

  return (
    <section className="topbar">
      <div className="topbar-main">
        <div className="control-group">
          <button
            type="button"
            className="icon-button"
            aria-label="Upload tracks"
            title="Upload tracks"
            onClick={() => fileInputRef.current?.click()}
          >
            <CloudUpload aria-hidden="true" size={18} strokeWidth={2.2} />
          </button>
          {tracksCount > 0 ? (
            <button
              type="button"
              className="icon-button danger-button"
              aria-label="Clear tracks"
              title="Clear tracks"
              onClick={onClearTracks}
            >
              <Trash2 aria-hidden="true" size={18} strokeWidth={2.2} />
            </button>
          ) : null}
        </div>
        <button
          type="button"
          className="icon-button view-mode-button"
          aria-label={`Current view: ${viewModeLabel}. Switch to ${nextViewMode.toUpperCase()} view`}
          title={`Switch to ${nextViewMode.toUpperCase()} view`}
          onClick={() => onViewModeChange(nextViewMode)}
        >
          {viewModeLabel}
        </button>
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        multiple
        onChange={(event) => {
          if (event.target.files) {
            onFilesSelected(event.target.files);
            event.target.value = "";
          }
        }}
      />
      {activeTracks.map((track) => (
        <span className="skeleton skeleton-chip" key={track.id} aria-label={`${track.title} processing`} />
      ))}
      {message ? <span className="status-pill">{message}</span> : null}
    </section>
  );
}
