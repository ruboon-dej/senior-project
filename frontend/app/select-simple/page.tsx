"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import ImagePlaceholder from "@/components/ImagePlaceholder";

export default function SelectSimplePage() {
  const router = useRouter();

  return (
    <div className="max-w-2xl mx-auto px-4 py-16 text-center">
      <h1 className="text-xl font-medium mb-10">Select your desired translation</h1>

      <div className="flex flex-col sm:flex-row justify-center gap-6">
        <button
          onClick={() => router.push("/character-select")}
          className="flex-1 rounded-2xl border bg-white p-6 shadow-sm hover:shadow-md hover:border-brand-500 transition"
        >
          {/* TODO: swap in the real icon/image for this option */}
          <ImagePlaceholder label="Thai → Thai SL" className="w-full h-32 mb-3" />
          <div className="font-medium">Thai → Thai SL</div>
        </button>

        <button
          onClick={() => router.push("/translate/thsl-to-thai")}
          className="flex-1 rounded-2xl border bg-white p-6 shadow-sm hover:shadow-md hover:border-brand-500 transition"
        >
          <ImagePlaceholder label="Thai SL → Thai" className="w-full h-32 mb-3" />
          <div className="font-medium">Thai SL → Thai</div>
        </button>
      </div>

    </div>
  );
}
