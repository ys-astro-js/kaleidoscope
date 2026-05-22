import type { RefObject, SyntheticEvent } from "react";
import { Trash2 } from "lucide-react";
import type { Track } from "../types";

export type AudioDeck = "primary" | "secondary";

type Props = {
  selected: Track | null;
  activeDeck: AudioDeck;
  primaryAudioRef: RefObject<HTMLAudioElement | null>;
  secondaryAudioRef: RefObject<HTMLAudioElement | null>;
  onDelete: (track: Track) => void;
  onTimeUpdate: (deck: AudioDeck, event: SyntheticEvent<HTMLAudioElement>) => void;
};

export function NowPlaying({
  selected,
  activeDeck,
  primaryAudioRef,
  secondaryAudioRef,
  onDelete,
  onTimeUpdate,
}: Props) {
  return (
    <section className={selected ? "now-playing active" : "now-playing idle"}>
      <div>
        <p>{selected?.title ?? ""}</p>
        <small>{selected?.artist ?? selected?.filename ?? ""}</small>
      </div>
      {selected ? (
        <button
          type="button"
          className="icon-button danger-button"
          aria-label={`Delete ${selected.title}`}
          title="Delete selected track"
          onClick={() => onDelete(selected)}
        >
          <Trash2 aria-hidden="true" size={17} strokeWidth={2.4} />
        </button>
      ) : null}
      <audio
        ref={primaryAudioRef}
        className={activeDeck === "primary" ? "active" : ""}
        controls={activeDeck === "primary"}
        onTimeUpdate={(event) => onTimeUpdate("primary", event)}
      />
      <audio
        ref={secondaryAudioRef}
        className={activeDeck === "secondary" ? "active" : ""}
        controls={activeDeck === "secondary"}
        onTimeUpdate={(event) => onTimeUpdate("secondary", event)}
      />
    </section>
  );
}
