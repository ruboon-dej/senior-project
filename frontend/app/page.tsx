"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import ImagePlaceholder from "@/components/ImagePlaceholder";

type Direction = "thai-to-thsl" | "thsl-to-thai";

export default function HomePage() {
  const router = useRouter();
  const [direction, setDirection] = useState<Direction>("thai-to-thsl");
  const [dragOverSide, setDragOverSide] = useState<"left" | "right" | null>(
    null
  );

  const left = direction === "thai-to-thsl" ? "Thai" : "Thai SL";
  const right = direction === "thai-to-thsl" ? "Thai SL" : "Thai";

  function handleSwap() {
    setDirection((prev) =>
      prev === "thai-to-thsl" ? "thsl-to-thai" : "thai-to-thsl"
    );
  }

  function handleConfirm() {
    if (direction === "thai-to-thsl") {
      router.push("/character-select");
    } else {
      router.push("/translate/thsl-to-thai");
    }
  }

  // Dragging either box onto the other swaps the direction, same as the
  // swap button — this just gives a drag-based way to trigger it.
  function handleDragStart(e: React.DragEvent, side: "left" | "right") {
    e.dataTransfer.setData("text/plain", side);
    e.dataTransfer.effectAllowed = "move";
  }

  function handleDrop(e: React.DragEvent, side: "left" | "right") {
    e.preventDefault();
    const sourceSide = e.dataTransfer.getData("text/plain");
    if (sourceSide && sourceSide !== side) {
      handleSwap();
    }
    setDragOverSide(null);
  }

  return (
    <div className="max-w-2xl mx-auto px-4 py-16 text-center">
      <h1 className="text-xl font-medium mb-2">Select your desired translation</h1>
      <p className="text-xs text-gray-400 mb-10">
        Drag one box onto the other, or use the swap button, to change direction
      </p>

      <div className="flex items-center justify-center gap-6 mb-8">
        <div
          draggable
          onDragStart={(e) => handleDragStart(e, "left")}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOverSide("left");
          }}
          onDragLeave={() => setDragOverSide(null)}
          onDrop={(e) => handleDrop(e, "left")}
          className={`flex flex-col items-center gap-2 cursor-grab active:cursor-grabbing rounded-xl transition ${
            dragOverSide === "left" ? "ring-2 ring-brand-400" : ""
          }`}
        >
          {/* TODO: swap in the real icon/character image for each language */}
          <ImagePlaceholder label={left} className="w-32 h-32 pointer-events-none" />
          <span className="text-sm text-gray-500">{left}</span>
        </div>

        {/* Fixed direction — always points from left box to right box */}
        <span className="text-2xl text-gray-400">&rarr;</span>

        <div
          draggable
          onDragStart={(e) => handleDragStart(e, "right")}
          onDragOver={(e) => {
            e.preventDefault();
            setDragOverSide("right");
          }}
          onDragLeave={() => setDragOverSide(null)}
          onDrop={(e) => handleDrop(e, "right")}
          className={`flex flex-col items-center gap-2 cursor-grab active:cursor-grabbing rounded-xl transition ${
            dragOverSide === "right" ? "ring-2 ring-brand-400" : ""
          }`}
        >
          <ImagePlaceholder label={right} className="w-32 h-32 pointer-events-none" />
          <span className="text-sm text-gray-500">{right}</span>
        </div>
      </div>

      <div className="flex items-center justify-center gap-3">
        <button
          onClick={handleSwap}
          title="Swap direction"
          className="w-11 h-11 rounded-full border flex items-center justify-center bg-white hover:bg-gray-50"
        >
          &#8646;
        </button>
        <button
          onClick={handleConfirm}
          className="px-6 py-2.5 rounded-full bg-brand-500 text-white text-sm font-medium hover:bg-brand-600"
        >
          Confirm
        </button>
      </div>

    </div>
  );
}