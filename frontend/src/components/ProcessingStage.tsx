import type { Track } from "../types";

type Props = {
  tracks: Track[];
};

export function ProcessingStage({ tracks }: Props) {
  if (tracks.length === 0) {
    return null;
  }

  return (
    <section className="skeleton-stage" aria-label="processing tracks">
      {tracks.map((track) => (
        <div className="skeleton skeleton-art" key={track.id} />
      ))}
    </section>
  );
}
