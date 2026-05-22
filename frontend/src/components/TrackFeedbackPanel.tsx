import { ThumbsDown, ThumbsUp } from "lucide-react";
import { artUrl } from "../api";
import type { FeedbackLabel, Track } from "../types";

export type SimilarTrackEntry = {
  score: number;
  track: Track;
};

type Props = {
  entries: SimilarTrackEntry[];
  onFeedback: (track: Track, label: FeedbackLabel) => void;
};

export function TrackFeedbackPanel({ entries, onFeedback }: Props) {
  if (entries.length === 0) {
    return null;
  }

  return (
    <section className="feedback-panel" aria-label="similarity feedback">
      <h2>Similar tracks</h2>
      {entries.map(({ score, track }) => (
        <div className="feedback-line" key={track.id}>
          <div className="track-summary">
            <img className="panel-art" src={artUrl(track.id)} alt="" loading="lazy" />
            <div>
              <strong>{track.title}</strong>
              <span>{Math.round(score * 100)}%</span>
            </div>
          </div>
          <div className="feedback-actions">
            <button
              type="button"
              className="with-icon"
              onClick={() => onFeedback(track, "similar")}
            >
              <ThumbsUp aria-hidden="true" size={14} strokeWidth={2.4} />
              Similar
            </button>
            <button
              type="button"
              className="with-icon"
              onClick={() => onFeedback(track, "not_similar")}
            >
              <ThumbsDown aria-hidden="true" size={14} strokeWidth={2.4} />
              Not similar
            </button>
          </div>
        </div>
      ))}
    </section>
  );
}
