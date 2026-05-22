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
    <section
      className="feedback-panel feedback-rail segment-panel"
      aria-label={`segment ${displaySegmentIndex + 1} similar segments`}
    >
      {entries.map(({ segmentIndex, startSeconds, track }) => (
        <button
          type="button"
          className="feedback-art-button"
          key={`${track.id}:${segmentIndex}`}
          aria-label={`${track.title}, segment ${segmentIndex + 1}`}
          title={`${track.title} · Segment ${segmentIndex + 1}`}
          onClick={() => onPlayTrack(track, startSeconds)}
        >
          <img src={artUrl(track.id)} alt="" loading="lazy" draggable={false} />
        </button>
      ))}
    </section>
  );
}
