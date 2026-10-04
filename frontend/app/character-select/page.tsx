"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

const ROW_COLORS = [
  "bg-pink-100",
  "bg-amber-100",
  "bg-yellow-100",
  "bg-rose-100",
  "bg-orange-100",
];

const CHARACTERS = [
  { id: "1", name: "Character 1" },
  { id: "2", name: "Character 2" },
  { id: "3", name: "Character 3" },
  { id: "4", name: "Character 4" },
  { id: "5", name: "Character 5" },
];

export default function CharacterSelectPage() {
  const router = useRouter();
  const [selectedId, setSelectedId] = useState(CHARACTERS[0].id);

  function handleConfirm() {
    router.push(`/translate/thai-to-thsl?character=${selectedId}`);
  }

  return (
    // TODO: replace bg-gray-200 with the real background image, e.g.
    // className="bg-[url('/character-select-bg.png')] bg-cover bg-center"
    <div className="min-h-[calc(100vh-57px)] bg-gray-200 relative overflow-hidden">
      {/* Top-right menu button (placeholder — image-based icon later) */}
      <button
        title="Menu (placeholder)"
        className="absolute top-6 right-6 z-10 w-11 h-11 rounded-full bg-white shadow flex items-center justify-center text-lg"
      >
        &#9776;
      </button>

      {/* The card floats over the background, leaving room around it */}
      <div className="relative w-[min(90%,42rem)] h-[80vh] mt-10 ml-10">
        {/* Top-left button, overlapping the card's corner */}
        <button
          onClick={() => router.push("/")}
          className="absolute -top-5 -left-5 z-10 w-12 h-12 rounded-full bg-white shadow-lg border flex items-center justify-center text-lg"
          aria-label="Back"
        >
          &larr;
        </button>

        <div className="w-full h-full bg-white rounded-3xl shadow-xl p-5 flex flex-col">
          <h1 className="text-center font-medium mb-4 mt-2">
            Select your character
          </h1>

          {/* Scrollable list of large picture-style rows */}
          <div className="flex-1 overflow-y-auto flex flex-col gap-3 pr-2 custom-scrollbar">
            {CHARACTERS.map((char, i) => (
              <button
                key={char.id}
                onClick={() => setSelectedId(char.id)}
                className={`relative shrink-0 h-24 rounded-2xl border-2 flex items-center justify-center text-center transition ${
                  ROW_COLORS[i % ROW_COLORS.length]
                } ${
                  selectedId === char.id
                    ? "border-brand-500 ring-2 ring-brand-200"
                    : "border-transparent hover:border-gray-300"
                }`}
              >
                {/* TODO: replace this whole block with a single character
                    image (character + name baked into one picture) */}
                <span className="font-medium text-gray-700">{char.name}</span>
                {selectedId === char.id && (
                  <span className="absolute top-2 right-2 text-brand-500 text-sm">
                    ✓
                  </span>
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Confirm button, overlapping the card's bottom-right corner */}
        <button
          onClick={handleConfirm}
          className="absolute -bottom-5 -right-5 px-7 py-3 rounded-full bg-brand-500 text-white text-sm font-medium shadow-lg hover:bg-brand-600"
        >
          Confirm
        </button>
      </div>
    </div>
  );
}