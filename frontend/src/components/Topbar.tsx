import { useRef } from "react";
import { CloudUpload, Repeat2, Trash2 } from "lucide-react";
import type { SimilarityMode, Track, ViewMode } from "../types";

type Props = {
  tracksCount: number;
  activeTracks: Track[];
  message: string;
  viewMode: ViewMode;
  similarityMode: SimilarityMode;
  isAutoplayActive: boolean;
  onViewModeChange: (mode: ViewMode) => void;
  onTrackModeSelect: () => void;
  onSegmentModeSelect: () => void;
  onAutoplayToggle: () => void;
  onClearTracks: () => void;
  onFilesSelected: (files: FileList) => void;
};

export function Topbar({
  tracksCount,
  activeTracks,
  message,
  viewMode,
  similarityMode,
  isAutoplayActive,
  onViewModeChange,
  onTrackModeSelect,
  onSegmentModeSelect,
  onAutoplayToggle,
  onClearTracks,
  onFilesSelected,
}: Props) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

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
        <div className="mode-toggle" role="group" aria-label="view mode">
          <button
            type="button"
            className={viewMode === "2d" ? "active" : ""}
            onClick={() => onViewModeChange("2d")}
          >
            2D
          </button>
          <button
            type="button"
            className={viewMode === "3d" ? "active" : ""}
            onClick={() => onViewModeChange("3d")}
          >
            3D
          </button>
        </div>
        <div className="mode-toggle" role="group" aria-label="similarity mode">
          <button
            type="button"
            className={similarityMode === "track" ? "active" : ""}
            onClick={onTrackModeSelect}
          >
            Track
          </button>
          <button
            type="button"
            className={similarityMode === "segment" ? "active" : ""}
            onClick={onSegmentModeSelect}
          >
            Segment
          </button>
        </div>
        <button
          type="button"
          className={
            isAutoplayActive
              ? "autoplay-toggle with-icon active"
              : "autoplay-toggle with-icon"
          }
          onClick={onAutoplayToggle}
        >
          <Repeat2 aria-hidden="true" size={16} strokeWidth={2.3} />
          Auto
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
