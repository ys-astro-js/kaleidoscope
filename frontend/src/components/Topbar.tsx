import { useRef } from "react";
import { CloudUpload, Fingerprint, Layers, Mic, Music, Trash2 } from "lucide-react";
import type { SimilarityMix, Track, ViewMode } from "../types";

type Props = {
  tracksCount: number;
  activeTracks: Track[];
  message: string;
  viewMode: ViewMode;
  similarityMix: SimilarityMix;
  onViewModeChange: (mode: ViewMode) => void;
  onSimilarityStemShareChange: (vocalShare: number) => void;
  onSimilaritySourceShareChange: (styleShare: number) => void;
  onClearTracks: () => void;
  onFilesSelected: (files: FileList) => void;
};

export function Topbar({
  tracksCount,
  activeTracks,
  message,
  viewMode,
  similarityMix,
  onViewModeChange,
  onSimilarityStemShareChange,
  onSimilaritySourceShareChange,
  onClearTracks,
  onFilesSelected,
}: Props) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const nextViewMode = viewMode === "2d" ? "3d" : "2d";
  const viewModeLabel = viewMode === "2d" ? "2D" : "3D";
  const stemTotal = similarityMix.vocals + similarityMix.instrumental;
  const vocalShare = stemTotal > 0 ? similarityMix.vocals / stemTotal : 0.5;
  const vocalPercent = Math.round(similarityMix.vocals * 100);
  const instrumentalPercent = Math.round(similarityMix.instrumental * 100);
  const sourceTotal = similarityMix.style + similarityMix.cover;
  const styleShare = sourceTotal > 0 ? similarityMix.style / sourceTotal : 0.65;
  const stylePercent = Math.round(similarityMix.style * 100);
  const coverPercent = Math.round(similarityMix.cover * 100);

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
        <div
          className="stem-mix-control"
          role="group"
          aria-label="Stem similarity mix"
          title="Whole similarity stays fixed at 50%; this slider splits the remaining 50%."
        >
          <Mic aria-hidden="true" size={16} strokeWidth={2.1} />
          <input
            type="range"
            min="0"
            max="100"
            step="1"
            value={Math.round(vocalShare * 100)}
            aria-label="Vocals and instrumental similarity split"
            onChange={(event) => onSimilarityStemShareChange(Number(event.currentTarget.value) / 100)}
          />
          <Music aria-hidden="true" size={16} strokeWidth={2.1} />
          <span className="stem-mix-value">{vocalPercent}/{instrumentalPercent}</span>
        </div>
        <div
          className="stem-mix-control source-mix-control"
          role="group"
          aria-label="Style and cover identity similarity mix"
          title="Final ranking mix between style similarity and Discogs-VINet cover identity."
        >
          <Layers aria-hidden="true" size={16} strokeWidth={2.1} />
          <input
            type="range"
            min="0"
            max="100"
            step="1"
            value={Math.round(styleShare * 100)}
            aria-label="Style and cover identity similarity split"
            onChange={(event) => (
              onSimilaritySourceShareChange(Number(event.currentTarget.value) / 100)
            )}
          />
          <Fingerprint aria-hidden="true" size={16} strokeWidth={2.1} />
          <span className="stem-mix-value">{stylePercent}/{coverPercent}</span>
        </div>
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
