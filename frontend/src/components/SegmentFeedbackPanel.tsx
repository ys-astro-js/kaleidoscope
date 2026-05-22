import { Play } from "lucide-react";
import { artUrl } from "../api";
import type { Track } from "../types";

export type SegmentTrackEntry = {
  score: number;
  segmentIndex: number;
  startSeconds: number;
  track: Track;
};

type Props = {
  displaySegmentIndex: number;
  entries: SegmentTrackEntry[];
  onPlayTrack: (track: Track, startSeconds: number) => void;
};

export function SegmentFeedbackPanel({
  displaySegmentIndex,
  entries,
  onPlayTrack,
}: Props) {
  if (entries.length === 0) {
    return null;
  }

  return (
    <section className="feedback-panel segment-panel" aria-label="similar segments">
      <h2>Segment {displaySegmentIndex + 1}</h2>
      {entries.map(({ score, segmentIndex, startSeconds, track }) => (
        <button
          type="button"
          className="segment-line"
          key={`${track.id}:${segmentIndex}`}
          onClick={() => onPlayTrack(track, startSeconds)}
        >
          <span className="track-summary">
            <img className="panel-art" src={artUrl(track.id)} alt="" loading="lazy" />
            <span>
              <strong>{track.title}</strong>
              <small>Segment {segmentIndex + 1}</small>
            </span>
          </span>
          <span className="segment-score">
            <Play aria-hidden="true" size={13} fill="currentColor" strokeWidth={0} />
            {Math.round(score * 100)}%
          </span>
        </button>
      ))}
    </section>
  );
}
