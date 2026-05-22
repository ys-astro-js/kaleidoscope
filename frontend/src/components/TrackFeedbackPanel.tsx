import { type CSSProperties, useRef, useState } from "react";
import { ChevronDown, ChevronUp, ThumbsDown, ThumbsUp } from "lucide-react";
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

type DragState = {
  pointerId: number;
  startY: number;
  track: Track;
};

const FEEDBACK_DRAG_THRESHOLD = 26;
const FEEDBACK_DRAG_LIMIT = 34;

export function TrackFeedbackPanel({ entries, onFeedback }: Props) {
  const dragStateRef = useRef<DragState | null>(null);
  const [draggingTrackId, setDraggingTrackId] = useState<string | null>(null);
  const [dragOffsetY, setDragOffsetY] = useState(0);

  if (entries.length === 0) {
    return null;
  }

  const finishDrag = (pointerId: number, endY: number) => {
    const dragState = dragStateRef.current;
    if (!dragState || dragState.pointerId !== pointerId) {
      return;
    }
    dragStateRef.current = null;
    setDraggingTrackId(null);
    setDragOffsetY(0);
    const deltaY = endY - dragState.startY;
    if (Math.abs(deltaY) < FEEDBACK_DRAG_THRESHOLD) {
      return;
    }
    onFeedback(dragState.track, deltaY < 0 ? "similar" : "not_similar");
  };

  return (
    <section className="feedback-panel feedback-rail" aria-label="similar tracks">
      {entries.map(({ score, track }) => (
        <button
          type="button"
          className={
            [
              "feedback-art-button",
              draggingTrackId === track.id ? "dragging" : "",
              draggingTrackId === track.id && dragOffsetY <= -FEEDBACK_DRAG_THRESHOLD
                ? "drag-similar"
                : "",
              draggingTrackId === track.id && dragOffsetY >= FEEDBACK_DRAG_THRESHOLD
                ? "drag-not-similar"
                : "",
            ].filter(Boolean).join(" ")
          }
          key={track.id}
          aria-label={`${track.title}, ${Math.round(score * 100)}% similar`}
          title={track.title}
          style={{
            "--drag-offset-y":
              draggingTrackId === track.id
                ? `${Math.max(-FEEDBACK_DRAG_LIMIT, Math.min(FEEDBACK_DRAG_LIMIT, dragOffsetY))}px`
                : "0px",
          } as CSSProperties}
          onPointerDown={(event) => {
            event.currentTarget.setPointerCapture(event.pointerId);
            dragStateRef.current = {
              pointerId: event.pointerId,
              startY: event.clientY,
              track,
            };
            setDraggingTrackId(track.id);
            setDragOffsetY(0);
          }}
          onPointerMove={(event) => {
            const dragState = dragStateRef.current;
            if (!dragState || dragState.pointerId !== event.pointerId) {
              return;
            }
            setDragOffsetY(event.clientY - dragState.startY);
          }}
          onPointerUp={(event) => finishDrag(event.pointerId, event.clientY)}
          onPointerCancel={(event) => {
            if (dragStateRef.current?.pointerId === event.pointerId) {
              dragStateRef.current = null;
              setDraggingTrackId(null);
              setDragOffsetY(0);
            }
          }}
        >
          <span className="feedback-choice feedback-choice-similar">
            <ChevronUp className="feedback-choice-chevron" aria-hidden="true" size={20} strokeWidth={2.5} />
            <ThumbsUp className="feedback-choice-thumb" aria-hidden="true" size={18} strokeWidth={2.5} />
          </span>
          <span className="feedback-art-drag">
            <img src={artUrl(track.id)} alt="" loading="lazy" draggable={false} />
            <span className="feedback-art-overlay">
              <span className="feedback-score">{Math.round(score * 100)}%</span>
            </span>
          </span>
          <span className="feedback-choice feedback-choice-not">
            <ChevronDown className="feedback-choice-chevron" aria-hidden="true" size={20} strokeWidth={2.5} />
            <ThumbsDown className="feedback-choice-thumb" aria-hidden="true" size={18} strokeWidth={2.5} />
          </span>
        </button>
      ))}
    </section>
  );
}
