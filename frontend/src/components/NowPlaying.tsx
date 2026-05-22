import { type RefObject, type SyntheticEvent, useEffect, useRef, useState } from "react";
import {
  Disc3,
  MoreHorizontal,
  Pause,
  Play,
  Repeat2,
  SkipBack,
  SkipForward,
  Trash2,
  Waypoints,
} from "lucide-react";
import { artUrl } from "../api";
import type { SimilarityMode, Track } from "../types";

export type AudioDeck = "primary" | "secondary";

type Props = {
  selected: Track | null;
  activeDeck: AudioDeck;
  primaryAudioRef: RefObject<HTMLAudioElement | null>;
  secondaryAudioRef: RefObject<HTMLAudioElement | null>;
  similarityMode: SimilarityMode;
  isAutoplayActive: boolean;
  isPlaying: boolean;
  currentTime: number;
  currentDuration: number;
  canPlayNext: boolean;
  onTrackModeSelect: () => void;
  onSegmentModeSelect: () => void;
  onPrevious: () => void;
  onPlayPause: () => void;
  onNext: () => void;
  onAutoplayToggle: () => void;
  onSeek: (seconds: number) => void;
  onDelete: (track: Track) => void;
  onTimeUpdate: (deck: AudioDeck, event: SyntheticEvent<HTMLAudioElement>) => void;
};

export function NowPlaying({
  selected,
  activeDeck,
  primaryAudioRef,
  secondaryAudioRef,
  similarityMode,
  isAutoplayActive,
  isPlaying,
  currentTime,
  currentDuration,
  canPlayNext,
  onTrackModeSelect,
  onSegmentModeSelect,
  onPrevious,
  onPlayPause,
  onNext,
  onAutoplayToggle,
  onSeek,
  onDelete,
  onTimeUpdate,
}: Props) {
  const [showRemaining, setShowRemaining] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [seekActive, setSeekActive] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const duration = Number.isFinite(currentDuration) ? currentDuration : 0;
  const progress = duration > 0 ? Math.min(100, (currentTime / duration) * 100) : 0;
  const subtitle = selected?.album ?? selected?.artist ?? selected?.filename ?? "";

  useEffect(() => {
    if (!menuOpen) {
      return;
    }
    const closeMenu = (event: MouseEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) {
        setMenuOpen(false);
      }
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", closeMenu);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("mousedown", closeMenu);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [menuOpen]);

  return (
    <section className={selected ? "now-playing active" : "now-playing idle"}>
      <div className="player-controls-left">
        <button
          type="button"
          className="player-mode-button"
          aria-label="Toggle similarity mode"
          title="Toggle similarity mode"
          onClick={similarityMode === "track" ? onSegmentModeSelect : onTrackModeSelect}
        >
          {similarityMode === "track" ? (
            <Disc3 aria-hidden="true" size={18} strokeWidth={2.3} />
          ) : (
            <Waypoints aria-hidden="true" size={18} strokeWidth={2.3} />
          )}
        </button>
        <button
          type="button"
          className="icon-button"
          aria-label="Previous"
          title="Previous"
          disabled={!selected}
          onClick={onPrevious}
        >
          <SkipBack aria-hidden="true" size={18} strokeWidth={2.3} />
        </button>
        <button
          type="button"
          className="icon-button primary-play-button"
          aria-label={isPlaying ? "Pause" : "Play"}
          title={isPlaying ? "Pause" : "Play"}
          disabled={!selected}
          onClick={onPlayPause}
        >
          {isPlaying ? (
            <Pause aria-hidden="true" size={18} strokeWidth={2.5} />
          ) : (
            <Play aria-hidden="true" size={18} strokeWidth={2.5} />
          )}
        </button>
        <button
          type="button"
          className="icon-button"
          aria-label="Next"
          title="Next"
          disabled={!selected || !canPlayNext}
          onClick={onNext}
        >
          <SkipForward aria-hidden="true" size={18} strokeWidth={2.3} />
        </button>
        <button
          type="button"
          className={
            isAutoplayActive
              ? "icon-button autoplay-toggle active"
              : "icon-button autoplay-toggle"
          }
          aria-label="Toggle autoplay"
          title="Autoplay"
          disabled={!selected}
          onClick={onAutoplayToggle}
        >
          <Repeat2 aria-hidden="true" size={17} strokeWidth={2.3} />
        </button>
      </div>

      <div className={seekActive ? "player-main seeking" : "player-main"}>
        <div className="seek-overlay" aria-hidden="true" />
        <div className="player-track-info">
          {selected ? (
            <img className="player-art" src={artUrl(selected.id)} alt="" draggable={false} />
          ) : (
            <div className="player-art" aria-hidden="true" />
          )}
          <div className="player-copy">
            <p>{selected?.title ?? ""}</p>
            <small>{subtitle}</small>
          </div>
        </div>
        <div
          className="seek-control"
          onMouseEnter={() => setSeekActive(true)}
          onMouseLeave={() => setSeekActive(false)}
          onFocus={() => setSeekActive(true)}
          onBlur={() => setSeekActive(false)}
        >
          <div className="seek-track" aria-hidden="true">
            <span style={{ width: `${progress}%` }} />
          </div>
          <input
            type="range"
            min={0}
            max={Math.max(duration, 0.01)}
            step={0.01}
            value={Math.min(currentTime, Math.max(duration, 0))}
            disabled={!selected || duration <= 0}
            aria-label="Seek"
            onChange={(event) => onSeek(Number(event.currentTarget.value))}
          />
          <div className="seek-times" aria-hidden={!selected}>
            <span>{formatTime(currentTime)}</span>
            <button
              type="button"
              disabled={!selected || duration <= 0}
              onClick={() => setShowRemaining((current) => !current)}
            >
              {showRemaining ? `-${formatTime(Math.max(0, duration - currentTime))}` : formatTime(duration)}
            </button>
          </div>
        </div>
      </div>

      <div className="player-menu" ref={menuRef}>
        <button
          type="button"
          className="icon-button"
          aria-label="Track menu"
          title="Track menu"
          disabled={!selected}
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen((current) => !current)}
        >
          <MoreHorizontal aria-hidden="true" size={19} strokeWidth={2.3} />
        </button>
        {menuOpen && selected ? (
          <div className="player-menu-popover" role="menu">
            <button
              type="button"
              className="danger-menu-item"
              role="menuitem"
              onClick={() => {
                setMenuOpen(false);
                onDelete(selected);
              }}
            >
              <Trash2 aria-hidden="true" size={16} strokeWidth={2.3} />
              Delete track
            </button>
          </div>
        ) : null}
      </div>

      <audio
        ref={primaryAudioRef}
        className={activeDeck === "primary" ? "active" : ""}
        onTimeUpdate={(event) => onTimeUpdate("primary", event)}
      />
      <audio
        ref={secondaryAudioRef}
        className={activeDeck === "secondary" ? "active" : ""}
        onTimeUpdate={(event) => onTimeUpdate("secondary", event)}
      />
    </section>
  );
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "0:00";
  }
  const rounded = Math.floor(seconds);
  const minutes = Math.floor(rounded / 60);
  const remainingSeconds = rounded % 60;
  return `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`;
}
