import { type RefObject, type SyntheticEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  Disc3,
  Mic,
  MoreHorizontal,
  Music,
  Pause,
  Play,
  Repeat2,
  SkipBack,
  SkipForward,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  Waypoints,
  type LucideIcon,
} from "lucide-react";
import { SEGMENT_HOP_SECONDS } from "../autoplay";
import { artUrl, resolveAudioStem } from "../api";
import type { AudioStem, FeedbackLabel, SimilarityMode, Track } from "../types";

export type AudioDeck = "primary" | "secondary";
export type AudioStemRefs = Record<AudioStem, RefObject<HTMLAudioElement | null>>;
export type AudioDeckRefs = Record<AudioDeck, AudioStemRefs>;

type Props = {
  selected: Track | null;
  activeDeck: AudioDeck;
  audioRefs: AudioDeckRefs;
  similarityMode: SimilarityMode;
  isAutoplayActive: boolean;
  isPlaying: boolean;
  currentTime: number;
  currentDuration: number;
  canPlayNext: boolean;
  feedbackNotice: { id: number; label: FeedbackLabel } | null;
  playbackStem: AudioStem;
  onTrackModeSelect: () => void;
  onSegmentModeSelect: () => void;
  onStemChange: (stem: AudioStem) => void;
  onPrevious: () => void;
  onPlayPause: () => void;
  onNext: () => void;
  onAutoplayToggle: () => void;
  onSeek: (seconds: number) => void;
  onDelete: (track: Track) => void;
  onTimeUpdate: (
    deck: AudioDeck,
    stem: AudioStem,
    event: SyntheticEvent<HTMLAudioElement>
  ) => void;
  onFeedbackNoticeDone: (id: number) => void;
};

export function NowPlaying({
  selected,
  activeDeck,
  audioRefs,
  similarityMode,
  isAutoplayActive,
  isPlaying,
  currentTime,
  currentDuration,
  canPlayNext,
  feedbackNotice,
  playbackStem,
  onTrackModeSelect,
  onSegmentModeSelect,
  onStemChange,
  onPrevious,
  onPlayPause,
  onNext,
  onAutoplayToggle,
  onSeek,
  onDelete,
  onTimeUpdate,
  onFeedbackNoticeDone,
}: Props) {
  const [showRemaining, setShowRemaining] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
  const [seekActive, setSeekActive] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const duration = Number.isFinite(currentDuration) ? currentDuration : 0;
  const progress = duration > 0 ? Math.min(100, (currentTime / duration) * 100) : 0;
  const subtitle = selected?.album ?? selected?.artist ?? selected?.filename ?? "";
  const activeStem = selected ? resolveAudioStem(selected, playbackStem) : "original";
  const stemOptions: Array<{ stem: AudioStem; label: string; Icon: LucideIcon }> = [
    { stem: "original", label: "Original", Icon: Disc3 },
    { stem: "vocals", label: "Vocals", Icon: Mic },
    { stem: "instrumental", label: "Instrumental", Icon: Music },
  ];
  const segmentPieces = useMemo(() => {
    if (similarityMode !== "segment" || duration <= 0 || !selected) {
      return [];
    }
    const pieceCount = Math.ceil(duration / SEGMENT_HOP_SECONDS);
    return Array.from({ length: pieceCount }, (_, index) => {
      const start = index * SEGMENT_HOP_SECONDS;
      const pieceDuration = Math.min(SEGMENT_HOP_SECONDS, Math.max(0, duration - start));
      const filledSeconds = Math.min(Math.max(currentTime - start, 0), pieceDuration);
      return {
        duration: pieceDuration,
        fill: pieceDuration > 0 ? (filledSeconds / pieceDuration) * 100 : 0,
        start,
      };
    });
  }, [currentTime, duration, selected, similarityMode]);
  const hasSegmentPieces = segmentPieces.length > 0;

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
      {feedbackNotice ? (
        <div
          className="player-feedback-notice"
          key={feedbackNotice.id}
          role="status"
          aria-live="polite"
          onAnimationEnd={() => onFeedbackNoticeDone(feedbackNotice.id)}
        >
          {feedbackNotice.label === "similar" ? (
            <ThumbsUp aria-hidden="true" size={16} strokeWidth={2.4} />
          ) : (
            <ThumbsDown aria-hidden="true" size={16} strokeWidth={2.4} />
          )}
          <span>
            {feedbackNotice.label === "similar"
              ? "비슷함으로 표시됨"
              : "비슷하지 않음으로 표시됨"}
          </span>
        </div>
      ) : null}
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
        <div className="stem-switcher" role="group" aria-label="Audio stem">
          {stemOptions.map(({ stem, label, Icon }) => {
            const isAvailable = selected
              ? selected.available_stems.includes(stem)
              : stem === "original";
            return (
              <button
                type="button"
                className={activeStem === stem ? "active" : ""}
                aria-label={label}
                title={label}
                disabled={!selected || !isAvailable}
                key={stem}
                onClick={() => onStemChange(stem)}
              >
                <Icon aria-hidden="true" size={15} strokeWidth={2.35} />
              </button>
            );
          })}
        </div>
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
          <div className={hasSegmentPieces ? "seek-track segmented" : "seek-track"} aria-hidden="true">
            {hasSegmentPieces ? (
              segmentPieces.map((piece) => (
                <span
                  className="seek-segment-piece"
                  key={piece.start}
                  style={{ flexGrow: piece.duration }}
                >
                  <span className="seek-segment-fill" style={{ width: `${piece.fill}%` }} />
                </span>
              ))
            ) : (
              <span style={{ width: `${progress}%` }} />
            )}
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

      {(["primary", "secondary"] as const).flatMap((deck) =>
        stemOptions.map(({ stem }) => (
          <audio
            ref={audioRefs[deck][stem]}
            className={activeDeck === deck && activeStem === stem ? "active" : ""}
            key={`${deck}-${stem}`}
            preload="auto"
            onTimeUpdate={(event) => onTimeUpdate(deck, stem, event)}
          />
        ))
      )}
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
